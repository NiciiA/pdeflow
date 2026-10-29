"""Named capacity choices and a real bounded native-resolution resource probe."""
from dataclasses import replace,asdict
from pathlib import Path
import time
import torch
from pdeflow.runtime import ModelConfig,code_identity,synchronize,process_peak_memory_bytes,write_json

def capacity(name):
    choices={'small':ModelConfig(width=8,blocks=1),'medium':ModelConfig(width=16,blocks=1,name='multiscale_unet',levels=3,conditioning_width=32,coordinate_features=True),'large':ModelConfig(width=24,blocks=2,name='multiscale_unet',levels=4,conditioning_width=48,coordinate_features=True,activation_checkpointing=True)}
    if name not in choices:raise ValueError('Capacity presets: small, medium, large')
    return choices[name]

def probe(config,output):
    from pdeflow.data.physical import specification,solve_spec
    from pdeflow.models import TimeChannelLayout,make_model,architecture_metadata
    from pdeflow.generative import make_formulation
    from pdeflow.core import Conditioning
    from pdeflow.specialists import get_specialist
    from pdeflow.observations import sensor_observations
    from pdeflow.steering import ReconstructionEnergy
    config.validate();output=Path(output);output.mkdir(parents=True,exist_ok=False);device,dtype=config.runtime.resolve();torch.manual_seed(config.training.seed)
    report={'source':code_identity(),'configuration':config.to_dict(),'device':str(device),'native_grid':list(config.grid.shape),'quality_claim':False,'meaning':'One real CFM optimizer update, native forward/backward and endpoint observation/PDE gradient; not a trained high-resolution reconstruction','stages':{}}
    def stage(name,fn):
        synchronize(device);start=time.perf_counter();value=fn();synchronize(device);report['stages'][name]={'seconds':time.perf_counter()-start,'passed':True};write_json(output/'probe.json',report);return value
    try:
        spec,seconds=specification(config,'calibration',0);clean=stage('native_training_reference',lambda:solve_spec(config,spec,seconds))
        specialist=get_specialist(config.specialist);layout=TimeChannelLayout(len(clean.problem.fields),config.grid)
        transform=specialist.transform_type.fit(clean.values[None],[spec['id']]);model=make_model(config.model,layout,specialist.encoder).to(device=device,dtype=dtype)
        report['architecture']=architecture_metadata(model,config.model,layout)
        condition=Conditioning.from_problem(clean.problem,1,device=device,dtype=dtype)
        target=layout.pack(transform.encode(clean.values[None].to(device=device,dtype=dtype)))
        formulation=make_formulation(config.formulation);x,y,s=formulation.training_state(target,torch.Generator().manual_seed(17),torch.Generator().manual_seed(18))
        opt=torch.optim.Adam(model.parameters(),lr=config.training.learning_rate);before=next(model.parameters()).detach().clone()
        def update():
            opt.zero_grad();prediction=model(x,s,condition);loss=formulation.loss(prediction,y);loss.backward();opt.step();return float(loss.detach())
        report['training_loss']=stage('native_CFM_forward_backward_update',update)
        report['weights_changed']=not torch.equal(before,next(model.parameters()).detach());assert report['weights_changed']
        model.eval();[p.requires_grad_(False) for p in model.parameters()]
        observations=sensor_observations(clean.values[None],config.grid,min(5,__import__('math').prod(config.grid.shape)),19,.03,.1)
        energy=ReconstructionEnergy(observations,specialist.physics(clean.problem,condition),lambda a:transform.decode(layout.unpack(a)),.3,.001)
        def guide():
            state=x.detach().requires_grad_(True);v=model(state,s,condition);estimate=formulation.endpoint(state,s.reshape(-1,*([1]*(state.ndim-1))),v)
            value=energy(estimate);gradient=torch.autograd.grad(value.sum(),state)[0]
            assert torch.isfinite(gradient).all();return float(gradient.norm().detach())
        report['guidance_gradient_norm']=stage('native_endpoint_physical_guidance_gradient',guide)
        from pdeflow.training import InferenceBundle
        from pdeflow.sampling import sample
        bundle=InferenceBundle(model,transform,layout,config,{'probe_only':True,'source':code_identity()})
        inf=replace(config.inference,steps=2,particles=1,controller=None,fk=None,snapshots={'max_frames':3,'max_particles':1})
        candidate=stage('native_source_to_candidate_Heun',lambda:sample(bundle,clean.problem,observations,inf,'unguided'))
        report['native_candidate']={'shape':list(candidate.samples.shape),'steps':candidate.diagnostics['steps'],'ode_model_evaluations':candidate.diagnostics['ode_evaluations'],'snapshot_model_evaluations':candidate.diagnostics['snapshot_model_evaluations'],'quality':'one-update execution probe; no trained-resolution accuracy claim'}
        import numpy as np
        from pdeflow.runtime import atomic_write
        atomic_write(output/'native-candidate.npz',lambda f:np.savez_compressed(f,samples=candidate.samples.numpy()))
        atomic_write(output/'generation.npz',lambda f:np.savez_compressed(f,**candidate.generation_snapshots['arrays']))
        write_json(output/'generation.json',candidate.generation_snapshots['metadata'])
        report['process_peak_rss_bytes']=process_peak_memory_bytes()
        if device.type=='mps':report['mps_memory']={'allocated_bytes':torch.mps.current_allocated_memory(),'driver_bytes':torch.mps.driver_allocated_memory(),'scope':'point-in-time after probe; not peak'}
        if device.type=='cuda':report['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['state']='passed';write_json(output/'probe.json',report);return report
    except BaseException as error:
        report.update(state='failed',error=f'{type(error).__name__}: {error}',original_grid_preserved=True);write_json(output/'probe.json',report);raise
