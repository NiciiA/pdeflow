"""Evaluation-only staged oracle and exact-observation reconstruction diagnosis."""
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from pdeflow.runtime import file_digest,write_json,atomic_write,code_identity,digest
from pdeflow.core import GridSpec,ProblemSpec
from .contracts import grid_queries
from .inference import CoordinateBundle
from .conditioning import prepare
from .training import normalized_parameter
from .physics import get,invariant_drift
from .diagnostic_cases import load_case,sensor_truth
from .diagnostic_inventory import inventory,oracle_model,conditioning_distribution
from .diagnostic_metrics import field_metrics,sensor_metrics


def exact_sensors(bundle,z,p,known,f,rows):
    pred=torch.empty(len(z),len(rows),device=bundle.device)
    for name in get(('heat','wave')[int(f[0])]).state:
        indices=[i for i,r in enumerate(rows) if r['field']==name]
        if indices:
            q=torch.tensor([[rows[i]['x'],rows[i]['y'],rows[i]['time']] for i in indices],device=bundle.device)
            pred[:,indices]=bundle.decode(z,q,p,known,f,name)
    return pred.detach().cpu().numpy()


def _plot(root,fields,truth,p,times):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['reference',*fields];selected=sorted({0,min(1,len(times)-1),len(times)//2,len(times)-1})
    for quantity,reference in truth.items():
        arrays=[reference]+[a[quantity].mean(0) for a in fields.values()];lo=min(float(x.min()) for x in arrays);hi=max(float(x.max()) for x in arrays)
        for initial,indices in [(True,[0]),(False,selected)]:
            fig,axes=plt.subplots(len(indices),len(arrays),squeeze=False,figsize=(3.2*len(arrays),3.1*len(indices)),layout='constrained')
            for row,k in enumerate(indices):
                for col,(name,a) in enumerate(zip(names,arrays)):
                    origin=p.grid.domain.origins;length=p.grid.domain.lengths;dx=length[0]/a.shape[-2];dy=length[1]/a.shape[-1]
                    # Pixel centres are periodic nodes, including wrapped edge halves.
                    displayed=np.pad(a[k],((0,1),(0,1)),mode='wrap')
                    im=axes[row,col].imshow(displayed.T,origin='lower',extent=[origin[0]-dx/2,origin[0]+length[0]+dx/2,origin[1]-dy/2,origin[1]+length[1]+dy/2],vmin=lo,vmax=hi,cmap='RdBu_r',interpolation='nearest')
                    axes[row,col].set_xlim(origin[0],origin[0]+length[0]);axes[row,col].set_ylim(origin[1],origin[1]+length[1])
                    title={'reference':'Reference','b':'B: generated','b_true_parameter_oracle':'B: true parameter (oracle)','Fourier_t0_control':'Fourier control (t=0)','a1':'A1: true initial input','a2':'A2: true initial + parameter'}.get(name,name)
                    axes[row,col].set_title(f'{title}\n{quantity}, physical t={times[k]:.4g}',fontsize=10)
                    axes[row,col].set_xlabel('x');axes[row,col].set_ylabel('y')
            fig.colorbar(im,ax=axes.ravel().tolist(),label='Physical value; shared scale')
            fig.savefig(root/f'{"initial-state" if initial else "evolved-state"}-{quantity}.png',dpi=120);plt.close(fig)


def shuffle_within_quantity(rows,rng):
    result=[dict(r) for r in rows]
    for quantity in sorted({r['field'] for r in rows}):
        indices=[i for i,r in enumerate(rows) if r['field']==quantity]
        for i,j in zip(indices,rng.permutation(indices)):result[i]['value']=rows[j]['value']
    return result


def observation_ablations(bundle,case,problem,noise,solver,only=None):
    """Same noise, coefficients, ODE and queries; only observations vary."""
    c=bundle.config;rows=case['observations'];true=case['true_parameter_evaluation_only'];settings=case['inference'];dev=bundle.device
    # Small common query set for bounded ablations, separate from main evaluation.
    q=grid_queries(GridSpec((12,12),problem.grid.domain,(problem.grid.times[0],problem.grid.times[-1])))
    variants={'correct':rows,'permuted':list(reversed(rows))}
    variants['shuffled_values']=shuffle_within_quantity(rows,np.random.default_rng(194))
    variants['perturbed_values']=[{**r,'value':r['value']+.2*bundle.cp['normalizations'][problem.family][0 if r['field']=='u' else 1]['scale']} for r in rows]
    retained=set()
    for quantity in sorted({r['field'] for r in rows}):retained.update([i for i,r in enumerate(rows) if r['field']==quantity][::2])
    variants['nested_half']=[r for i,r in enumerate(rows) if i in retained]
    if bundle.cp['capabilities']['unconditional_trained']:variants['none']=[]
    if problem.family=='wave':variants.update({name+'_only':[r for r in rows if r['field']==name] for name in ('u','v') if any(r['field']==name for r in rows)})
    records={};base=None;heldout=case['heldout_observations_evaluation_only'];clean=sensor_truth(solver,rows,true,c.interval)
    if only is not None and ('correct' not in only or set(only)-set(variants)):raise ValueError('Selected ablations must include correct and supported variants')
    for label,observations in variants.items():
        if only is not None and label not in only:continue
        capture={}
        def observe(module,args,out):capture['observation_embedding']=out.detach().cpu()
        def velocity(module,args,out):
            if 'flow_velocity_at_s0' not in capture:capture['flow_velocity_at_s0']=out.detach().cpu()
        handles=[bundle.velocity.observation_encoder.register_forward_hook(observe),bundle.velocity.register_forward_hook(velocity)]
        try:
            samples,report=bundle.sample(problem,observations,q,initial_noise=noise,**settings)
        finally:
            for handle in handles:handle.remove()
        z=bundle.decoded_latent;capture['latent_initial']=z;capture['field']=torch.tensor(samples)
        sensors=exact_sensors(bundle,z.to(dev),bundle.parameter.to(dev),bundle.known.to(dev),bundle.families.to(dev),rows)
        held=exact_sensors(bundle,z.to(dev),bundle.parameter.to(dev),bundle.known.to(dev),bundle.families.to(dev),heldout)
        if base is None:base=capture
        records[label]={'count':len(observations),'differences_from_correct':{k:{'rms':float((value-base[k]).double().square().mean().sqrt()),'maximum_absolute':float((value-base[k]).abs().max())} for k,value in capture.items()},
                        'conditioning_sensor_set_metrics':sensor_metrics(sensors,rows,clean),'heldout_sensor_metrics':sensor_metrics(held,heldout,[r['value'] for r in heldout]),'inferred_parameters':report['inferred_parameters']}
    # Padded invalid rows must not affect context or flow; no reference is supplied.
    packed=prepare(c,bundle.cp['normalizations'],problem,rows,len(noise),dev);obs=packed.observations
    padded=tuple(torch.cat([x,torch.full_like(x[:,:3],float('nan')) if x.is_floating_point() else torch.zeros_like(x[:,:3])],1) for x in obs)
    with torch.no_grad():
        graph=bundle.ae.graph_encoder(packed.families)
        a=bundle.velocity.condition(packed.parameter,obs,packed.known,packed.families,graph)
        b=bundle.velocity.condition(packed.parameter,padded,packed.known,packed.families,graph)
    records['masked_padding']={'context_maximum_absolute_difference':float((a[0]-b[0]).abs().max()),'all_finite':bool(torch.isfinite(b[0]).all())}
    return {'variants':records,'unsupported_no_observations':None if bundle.cp['capabilities']['unconditional_trained'] else 'Checkpoint declares no unconditional training; no-observation inference rejected rather than forced',
            'controls':'Identical explicit noise, known-status, queries, solver/steps and checkpoint; only observation rows differ. Value shuffling and count reduction preserve quantity groups; no u/v unit mixing. Ablation output grid12² at first/last times, not a throughput or main-grid accuracy comparison.'}


def diagnose(checkpoint,case_path,output,device='cpu',stages=('a1','a2','b'),evolution=None,representation=None,ablations=False,integration_steps=()):
    started=time.perf_counter();root=Path(output)
    if root.exists():raise FileExistsError('Choose a fresh diagnostic report directory')
    if not set(stages)<=set(('a1','a2','b')) or not stages:raise ValueError('Diagnostic stages: a1,a2,b')
    case_root=Path(case_path);case_root=case_root.parent if case_root.is_file() else case_root
    case,p,solver,context,noise=load_case(case_root)
    if file_digest(checkpoint)!=case['checkpoint_sha256']:raise ValueError('Diagnostic case belongs to a different checkpoint; capture an explicit new comparison case')
    bundle=CoordinateBundle(checkpoint,device);bundle.checkpoint_path=str(Path(checkpoint).resolve());c=bundle.config;c.compatible(p)
    ae,lineage=oracle_model(bundle,evolution,representation)
    root.mkdir(parents=True);q=grid_queries(p.grid);shape=(len(p.grid.times),*p.grid.shape);names=get(p.family).state;parameter=get(p.family).parameter;true=case['true_parameter_evaluation_only'];rows=case['observations'];heldout=case['heldout_observations_evaluation_only']
    truth={name:solver.evaluate(q.numpy(),true,c.interval,name).reshape(shape) for name in names};fields={};initial_fields={};reports={};raw={};clean=sensor_truth(solver,rows,true,c.interval)
    base={'case_identity':case['case_identity'],'conditioning_distribution':conditioning_distribution(c,p,rows),'checkpoint':inventory(checkpoint),'oracle_lineage':lineage,'reference_label':case['reference_label'],'distribution_assessment':case['distribution_assessment'],
          'reference_invariants':[{'time':t,**solver.invariants(t,true)} for t in p.grid.times],'source':code_identity(),'inference':case['inference'],'selection':case['selection'],'device':str(bundle.device),'precision':'float32; eval mode; dropout disabled','times':list(p.grid.times)}
    def evaluate(label,values,z,pn,known,f,description):
        initial_fields[label]={name:(bundle.ae.decode_state(z,q[:np.prod(p.grid.shape),:2].to(bundle.device),j,f)*bundle.cp['normalizations'][p.family][j]['scale']+bundle.cp['normalizations'][p.family][j]['mean']).detach().cpu().numpy().reshape(len(z),1,*p.grid.shape) for j,name in enumerate(names)}
        fields[label]=values;raw.update({label+'_'+name:a for name,a in values.items()})
        quantities={name:field_metrics(values[name],truth[name],p.grid.times,c,name,bundle.cp['normalizations'][p.family][j]) for j,name in enumerate(names)}
        sensor=exact_sensors(bundle,z,pn,known,f,rows);held=exact_sensors(bundle,z,pn,known,f,heldout)
        physical_parameters=bundle.ae.decode_parameter(z,pn,known,True,f).detach().cpu().numpy()
        reports[label]={'meaning':description,'physics_invariants':invariant_drift(values['u'],values.get('v'),p.grid.times,c.lengths,physical_parameters),'quantities':quantities,'conditioning_sensor_metrics':sensor_metrics(sensor,rows,clean),'heldout_sensor_metrics':sensor_metrics(held,heldout,[r['value'] for r in heldout]),
                        'wave_combined_normalized_mse':float(np.mean([quantities[name]['ensemble_mean']['uniform_physical']['mse']/bundle.cp['normalizations'][p.family][j]['scale']**2 for j,name in enumerate(names)])) if p.family=='wave' else None}
        raw[label+'_conditioning_sensor_predictions']=sensor;raw[label+'_heldout_sensor_predictions']=held
    if ae is not None and any(s in stages for s in ('a1','a2')):
        f=torch.tensor([int(p.family=='wave')],device=bundle.device);known=torch.ones(1,dtype=torch.bool,device=bundle.device);pn=normalized_parameter(c,torch.tensor([true],device=bundle.device),f)
        xy=torch.tensor(context,device=bundle.device,dtype=torch.float32);values=np.stack([solver.evaluate(np.c_[context,np.zeros(len(context))],true,c.interval,name) for name in names],-1)
        if p.family=='heat':values=np.c_[values,np.zeros(len(context))]
        for j,norm in enumerate(bundle.cp['normalizations'][p.family]):values[:,j]=(values[:,j]-norm['mean'])/norm['scale']
        if p.family=='heat':values[:,1]=0
        with torch.no_grad():
            z=ae.encode(xy[None],torch.tensor(values,dtype=torch.float32,device=bundle.device)[None],pn,f)
            previous=bundle.ae;bundle.ae=ae
            try:
                if 'a1' in stages:
                    decoded={name:(ae.decode_state(z,q[:np.prod(p.grid.shape),:2].to(bundle.device),j,f)*bundle.cp['normalizations'][p.family][j]['scale']+bundle.cp['normalizations'][p.family][j]['mean']).cpu().numpy().reshape(len(z),1,*p.grid.shape) for j,name in enumerate(names)}
                    # A1 is ONLY t=0. Do not score a stationary repetition against dynamics.
                    initial={name:field_metrics(a,solver.frame(0.,true,c.interval,p.grid.shape,name)[None],[0.],c,name,bundle.cp['normalizations'][p.family][j]) for j,(name,a) in enumerate(decoded.items())}
                    reports['a1']={'meaning':'Oracle complete initial state E(X0), spatial D at t=0; not inference from observations','quantities':initial}
                    raw.update({'a1_'+name:a[:,:1] for name,a in decoded.items()});initial_fields['a1']=decoded
                if 'a2' in stages:
                    decoded={name:bundle.decode(z,q.to(bundle.device),pn,known,f,name).cpu().numpy().reshape(1,*shape) for name in names}
                    evaluate('a2',decoded,z,pn,known,f,'Oracle true initial u0/v0 and true parameter, learned T and D; not deployable inverse performance')
            finally:bundle.ae=previous
    else:
        for stage in ('a1','a2'):
            if stage in stages:reports[stage]={'status':'unavailable','reason':lineage['reason']}
    if 'b' in stages:
        samples,report=bundle.sample(p,rows,q,initial_noise=noise,**case['inference']);z=bundle.decoded_latent.to(bundle.device);pn=bundle.parameter.to(bundle.device);known=bundle.known.to(bundle.device);f=bundle.families.to(bundle.device)
        decoded={'u':samples.reshape(len(noise),*shape)}
        if p.family=='wave':decoded['v']=bundle.velocity_samples.reshape(len(noise),*shape)
        with torch.no_grad():evaluate('b',decoded,z,pn,known,f,'Production generation from exactly saved observations; ensemble mean and every sample reported')
        reports['b']['production_report']=report
        if case.get('original'):
            with np.load(case_root/'original-result.npz',allow_pickle=False) as saved:
                reports['b']['replay_maximum_absolute_difference']=float(np.max(abs(samples-saved['samples'])))
                reports['b']['query_coordinates_match_saved']=bool(np.array_equal(q.numpy(),saved['queries']))
        if not p.known_parameters and c.joint['known_probability']>0:
            oracle=canonical_known(p,true);paired,paired_report=bundle.sample(oracle,rows,q,initial_noise=noise,**case['inference']);values={'u':paired.reshape(len(noise),*shape)}
            if p.family=='wave':values['v']=bundle.velocity_samples.reshape(len(noise),*shape)
            with torch.no_grad():evaluate('b_true_parameter_oracle',values,bundle.decoded_latent.to(bundle.device),bundle.parameter.to(bundle.device),bundle.known.to(bundle.device),bundle.families.to(bundle.device),'Same measurements/noise, true physical parameter explicitly disclosed: oracle diagnostic')
            reports['b_true_parameter_oracle']['production_report']=paired_report
        if integration_steps:base['integration_probe']=integration_probe(bundle,case,p,noise,solver,integration_steps)
        if ablations:base['conditioning_ablations']=observation_ablations(bundle,case,p,noise,solver)
    from .diagnostic_information import initial_fourier_check
    initial_fit,information=initial_fourier_check(c,p,rows,solver,true);base['initial_information_check']=information
    if initial_fit is not None and information['full_column_rank']:
        initial_fields['Fourier_t0_control']={'u':initial_fit[None,None]};reports['Fourier_t0_control']={'meaning':information['limits']}
    # Keep complete initial-state plots distinct from evolved results.
    _plot(root,fields,truth,p,list(p.grid.times))
    if initial_fields:
        directory=root/'oracle-initial';directory.mkdir()
        initial_truth={name:solver.frame(0.,true,c.interval,p.grid.shape,name)[None] for name in names}
        _plot(directory,initial_fields,initial_truth,p,[0.])
        for label,values in initial_fields.items():
            reports[label]['initial_state_metrics']={name:field_metrics(a,initial_truth[name],[0.],c,name,bundle.cp['normalizations'][p.family][j]) for j,(name,a) in enumerate(values.items())}
            raw.update({label+'_initial_'+name:a for name,a in values.items()})
    base.update(stages=reports,seconds=time.perf_counter()-started,limits=['Missing matching encoder means A1/A2 cannot be attributed for that export','An oracle is not production reconstruction','Tiny ensembles do not establish calibrated uncertainty','Initial-state failure can implicate A1, B or their interface; it does not prove only A1 is wrong'])
    raw.update({'reference_'+name:a for name,a in truth.items()});raw['queries']=q.numpy();atomic_write(root/'fields.npz',lambda out:np.savez_compressed(out,**raw));write_json(root/'diagnosis.json',base)
    return base


def canonical_known(problem,parameter):
    phys=get(problem.family)
    return replace(problem,known_parameters={phys.parameter:parameter},unknown_parameters=())


def integration_probe(bundle,case,problem,noise,solver,counts=(4,8,16)):
    if not 2<=len(counts)<=4 or any(type(n)!=int or not 1<=n<=64 for n in counts):raise ValueError('Use2..4 bounded integration counts in1..64')
    q=grid_queries(GridSpec((12,12),problem.grid.domain,(problem.grid.times[0],problem.grid.times[-1])));results=[];fields=[];latents=[]
    reference=solver.evaluate(q.numpy(),case['true_parameter_evaluation_only'],bundle.config.interval,'u')
    for n in counts:
        settings={**case['inference'],'steps':n};a,r=bundle.sample(problem,case['observations'],q,initial_noise=noise,**settings);fields.append(a);latents.append(bundle.decoded_latent.clone())
        results.append({'steps':n,'nfe':r['flow_solver']['nfe'],'ensemble_mean_u_mse':float(np.mean((a.mean(0)-reference)**2)),'sensor_metrics':sensor_metrics(r['observation_consistency']['predictions_per_sample'],case['observations']) if case['observations'] else None})
    high=int(np.argmax(counts))
    for i,r in enumerate(results):r.update(field_rms_difference_from_highest_steps=float(np.sqrt(np.mean((fields[i]-fields[high])**2))),latent_rms_difference_from_highest_steps=float((latents[i]-latents[high]).double().square().mean().sqrt()))
    return {'settings':'Identical noise/observations/coefficients;12² first+last-time evaluation; step count only changes. Highest count is a convergence reference, not ground truth.','results':results}
