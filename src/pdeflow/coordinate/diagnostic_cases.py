"""Immutable replay cases: full references remain evaluation-only."""
from pathlib import Path
import numpy as np
import torch
from pdeflow.runtime import read_json,write_json,file_digest,code_identity,atomic_write,digest
from pdeflow.core import ProblemSpec,GridSpec,DomainSpec
from .training import read_checkpoint
from .contracts import grid_queries,validate_rows
from .physics import get,problem as canonical_problem
from .dataset import example,recipe
from pdeflow.numerics.dynamic import SpectralState


def load_case(path):
    root=Path(path);root=root.parent if root.is_file() else root
    case=read_json(root/'case.json')
    if case.get('schema')!='operator_diagnostic_case_v1':raise ValueError('Unsupported diagnostic case')
    metadata={k:v for k,v in case.items() if k!='case_identity'}
    if digest(metadata)!=case['case_identity']:raise ValueError('Diagnostic case metadata checksum mismatch')
    for name,sha in case['files'].items():
        if Path(name).name!=name or file_digest(root/name)!=sha:raise ValueError('Diagnostic case file checksum mismatch: '+name)
    with np.load(root/'reference.npz',allow_pickle=False) as a:initial=a['initial'].copy();context=a['encoder_coordinates'].copy()
    p=ProblemSpec.from_dict(case['problem']);true=case['true_parameter_evaluation_only']
    solver=SpectralState(initial[...,0],initial[...,1] if p.family=='wave' else None,p.grid.domain.lengths,p.grid.domain.origins,p.family)
    with np.load(root/'noise.npz',allow_pickle=False) as a:noise=a['initial_flow_noise'].copy()
    return case,p,solver,context,noise


def sensor_truth(solver,rows,parameter,interval):
    result=np.empty(len(rows))
    for name in ('u','v'):
        ix=[i for i,r in enumerate(rows) if r['field']==name]
        if ix:
            q=np.array([[rows[i]['x'],rows[i]['y'],rows[i]['time']] for i in ix]);result[ix]=solver.evaluate(q,parameter,interval,name)
    return result


def capture(checkpoint,output,job=None,reference=None,validation_index=None,observations=None,problem=None,seed=91,steps=8,particles=4,resolution=32,training_observations=False):
    if validation_index is not None and (job or reference):raise ValueError('Choose a saved job/reference OR a validation realization')
    if training_observations and (validation_index is None or job or reference):raise ValueError('Training observations apply only to validation cases; saved measurements are never regenerated')
    cp,c=read_checkpoint(checkpoint,'flow');root=Path(output)
    if root.exists():raise FileExistsError('Use a fresh diagnostic case directory')
    if not 8<=resolution<=128 or not 1<=particles<=16 or not 1<=steps<=512:raise ValueError('Bound diagnostic grid8..128, particles1..16 and steps1..512')
    original=None;original_arrays=None;context=None
    if job:
        j=Path(job);request=read_json(j/'request.json');problem=ProblemSpec.from_dict(request['problem']);rows=request['rows'];settings=request['inference']
        if request['model_id']!=file_digest(checkpoint)[:32]:raise ValueError('Saved job used a different checkpoint')
        seed=settings['seed'];steps=settings['steps'];particles=settings['particles']
        if settings.get('correction'):raise ValueError('Capture uncorrected saved job first; corrected states need a separately declared comparison')
        if reference is None:
            report=read_json(j/'coordinate-result/comparison.json');reference=j.parent.parent/'data'/report['reference_id']
        original={'job':str(j.resolve()),'request':request,'job_record':read_json(j/'job.json'),'metrics':read_json(j/'coordinate-result/metrics.json')}
        with np.load(j/'coordinate-result/result.npz',allow_pickle=False) as a:original_arrays={k:a[k].copy() for k in a.files}
    elif validation_index is not None:
        if not 0<=validation_index<c.validation_count:raise ValueError('Validation index outside checkpoint split')
        item=example(c,'validation',validation_index);solver=item['solver'];true=item['parameter'];context=item['context'];family=item['family'];parameter=get(family).parameter
        grid=GridSpec((resolution,resolution),DomainSpec(c.lengths,c.origins),(0.,.01,.05,.1,.237,.5,.75,1.))
        problem=canonical_problem(grid,family,true);rng=np.random.default_rng(seed);rows=[]
        for i in range(min(48,c.observation_counts[1])):
            x,y=rng.random(2)*np.array(c.lengths)+np.array(c.origins);rows.append({'field':get(family).state[i%len(get(family).state)],'x':float(x),'y':float(y),'time':float(grid.times[i%len(grid.times)]),'value':0.,'sigma':0.})
        clean=sensor_truth(solver,rows,true,c.interval)
        rows=[{**r,'value':float(v)} for r,v in zip(rows,clean)]
        if training_observations:
            from .engine import observation_batch
            from .conditioning import physical_rows
            packed=observation_batch(c,[item],torch.Generator().manual_seed(seed),cp['normalizations'],torch.device('cpu'))
            rows=physical_rows(packed,cp['normalizations'],family)
        label={'type':'training-like validation','id':item['id'],'case':item['case'],'split':'validation','recipe':recipe(c)}
    if reference:
        from pdeflow.reference import load
        reference=Path(reference);reference=reference.parent if reference.is_file() else reference
        rp,values,manifest=load(reference);family=rp.family;true=rp.known_parameters[get(family).parameter]
        if tuple(rp.grid.domain.lengths)!=c.lengths or tuple(rp.grid.domain.origins)!=c.origins:raise ValueError('Reference domain differs from checkpoint')
        solver=SpectralState(values[0,0,0],values[0,1,0] if family=='wave' else None,c.lengths,c.origins,family)
        if not job:
            problem=ProblemSpec.from_dict(read_json(problem)) if isinstance(problem,(str,Path)) else problem
            problem=problem or canonical_problem(rp.grid,family,None)
            if observations is None:raise ValueError('A custom reference requires saved observations; never regenerate measurements silently')
            rows=read_json(observations) if isinstance(observations,(str,Path)) else observations
        label={'type':'user reference; distribution membership not assumed','reference_sha256':manifest['array_sha256'],'reference_id':manifest['id'],'specification':read_json(Path(reference)/'reference-problem.json')}
    elif validation_index is None:raise ValueError('Choose a saved job/reference or a validation realization')
    c.compatible(problem);validate_rows(c,rows,problem)
    if problem.family!=solver.family:raise ValueError('Reference and requested PDE family differ')
    times=list(problem.grid.times);shape=tuple(problem.grid.shape)
    if context is None:
        context=np.random.default_rng(seed+793).random((c.encoder_points,2))*np.asarray(c.lengths)+np.asarray(c.origins)
    heldout=[];rng=np.random.default_rng(seed+991)
    for i in range(64):
        xy=rng.random(2)*np.asarray(c.lengths)+np.asarray(c.origins);heldout.append({'field':get(problem.family).state[i%len(get(problem.family).state)],'x':float(xy[0]),'y':float(xy[1]),'time':float(times[i%len(times)]),'value':0.,'sigma':0.})
    clean=sensor_truth(solver,heldout,true,c.interval);heldout=[{**r,'value':float(v)} for r,v in zip(heldout,clean)]
    initial=np.stack([solver.initial,solver.velocity],-1);modes=np.meshgrid(*[np.fft.fftfreq(n)*n for n in solver.initial.shape],indexing='ij');outside=np.logical_or.reduce([abs(k)>recipe(c)['max_wave'] for k in modes]);power=abs(np.fft.fftn(solver.initial))**2
    ood={'initial_u_power_outside_training_band_fraction':float(power[outside].sum()/max(power.sum(),1e-30)),'training_max_wave':recipe(c)['max_wave'],
         'note':'Nonzero out-of-band power flags a distribution difference, not a proven cause. Heat spot widths and wave velocity distributions require separate interpretation.'}
    root.mkdir(parents=True);noise=torch.randn(particles,c.tokens,c.width,generator=torch.Generator().manual_seed(seed)).numpy()
    atomic_write(root/'reference.npz',lambda f:np.savez_compressed(f,initial=initial,encoder_coordinates=context));atomic_write(root/'noise.npz',lambda f:np.savez_compressed(f,initial_flow_noise=noise))
    if original_arrays:atomic_write(root/'original-result.npz',lambda f:np.savez_compressed(f,**original_arrays))
    case={'schema':'operator_diagnostic_case_v1','checkpoint_sha256':file_digest(checkpoint),'checkpoint':str(Path(checkpoint).resolve()),'source':code_identity(),
          'problem':problem.to_dict(),'reference_label':label,'distribution_assessment':ood,'true_parameter_evaluation_only':float(true),
          'observations':rows,'heldout_observations_evaluation_only':heldout,'inference':{'seed':seed,'steps':steps,'particles':particles,'solver':settings.get('solver','heun') if job else 'heun','chunk_size':settings.get('chunk_size',1024) if job else 1024},
          'observation_policy':'exact saved measurements' if job or reference else 'production_training_sampler' if training_observations else 'fixed_clean_grid','selection':'ensemble mean (playground display) and all individual samples','original':original,'files':{p.name:file_digest(p) for p in root.iterdir() if p.is_file()},
          'separation':'Physical time t, generative flow time s, wave velocity v=du/dt, flow velocity dZ/ds are distinct. Full initial state/true parameter/heldout values are evaluation only.'}
    case['case_identity']=digest(case);write_json(root/'case.json',case);return case
