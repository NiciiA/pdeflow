"""Fixed held-out cases, identical observations/noise, complete ensemble summaries."""
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from pdeflow.core import GridSpec
from pdeflow.runtime import RuntimeSpec,InferenceConfig,write_json,atomic_write,code_identity,file_digest,process_peak_memory_bytes
from pdeflow.training import load_inference,_read_checkpoint
from pdeflow.sampling import sample
from pdeflow.data.physical import specification,solve_spec
from pdeflow.reference.observations import ObservationSamplingSpec
from pdeflow.observations import sensor_observations
from pdeflow.workbench.observations import observation_rows
from .inference import CoordinateBundle
from .contracts import grid_queries
from .evaluation import interpolate_grid,error_summary,common_grid_diagnostics


def compare(coordinate,baseline,output,counts=(4,12),noises=(0.,.03),resolutions=(8,12),seeds=(91,92),cases=2,particles=3,steps=8,device='cpu'):
    root=Path(output)
    if root.exists():raise FileExistsError('Comparison destination exists; choose a fresh directory')
    if not 1<=cases<=16 or len(counts)*len(noises)*len(seeds)*cases>256 or any(n<1 or n>20000 for n in counts) or any(not np.isfinite(s) or s<0 for s in noises):raise ValueError('Comparison exceeds explicit bounded case/sensitivity budget')
    fixed=load_inference(baseline,runtime=RuntimeSpec(device=device));flex=CoordinateBundle(coordinate,device)
    if fixed.config.specialist!='heat':raise ValueError('This comparison supports the heat specialist only')
    if any(type(n)!=int or n<4 or n**fixed.config.grid.dimension*len(fixed.config.grid.times)>2_000_000 for n in resolutions):raise ValueError('Query resolutions exceed bounded physical-field budget')
    root.mkdir(parents=True);records=[];case_metadata=[]
    for case in range(cases):
        spec,seconds=specification(fixed.config,'validation',case);physical=solve_spec(fixed.config,spec,seconds);p=physical.problem
        flex.config.compatible(p);truth=physical.values[None]
        case_metadata.append({'id':spec['id'],'physical_specification':spec,'split':'validation','reference_grid':list(p.grid.shape),'reference_generation_seconds':physical.timings})
        for count in counts:
            for noise in noises:
                obs=sensor_observations(truth,p.grid,count,1000+case,float(noise));rows=observation_rows(obs,p)
                rows=[{**r,'sigma':float(noise)} for r in rows]
                for seed in seeds:
                    inference=InferenceConfig(steps=steps,particles=particles,seed=seed,controller=None)
                    started=time.perf_counter();result=sample(fixed,p,obs,inference,'observation_pde');fixed_seconds=time.perf_counter()-started
                    for resolution in resolutions:
                        grid=replace(p.grid,shape=(resolution,)*p.grid.dimension);qp=replace(p,grid=grid);q=grid_queries(grid)
                        generated,report=flex.sample(qp,rows,q,steps,particles,seed)
                        target=interpolate_grid(truth[0,0].numpy(),p.grid,q.numpy())
                        baseline_samples=np.stack([interpolate_grid(x[0].numpy(),p.grid,q.numpy()) for x in result.samples[0]])
                        row={'case_id':spec['id'],'spatial_sensors':count,'physical_times':len(p.grid.times),'total_scalar_observations':len(rows),'absolute_sigma':noise,'seed':seed,'ensemble_size':particles,'query_shape':list(grid.shape),'query_physical_times':grid.times,'reference_transfer':'periodic multilinear space/linear time of common held-out reference; approximate at new points, exact at saved nodes; no fine-scale truth claim','coordinate':{**error_summary(generated,target),**common_grid_diagnostics(generated,grid,p.known_parameters['diffusivity']),'seconds':report['seconds'],'memory':report['memory'],'observation_consistency':report['observation_consistency'],'native_coordinate_queries':True},'fixed':{**error_summary(baseline_samples,target),**common_grid_diagnostics(baseline_samples,grid,p.known_parameters['diffusivity']),'seconds_native_generation':fixed_seconds,'memory':result.diagnostics,'observation_rmse_against_noisy_per_sample':torch.sqrt(((obs.operator.apply(result.samples[0])-obs.values)**2).mean(1)).tolist(),'interpolation':'none' if grid==p.grid else 'fixed-grid baseline plus periodic multilinear spatial interpolation; not native flexibility'}}
                        shaped=(generated-target).reshape(particles,len(grid.times),-1);row['coordinate']['initial_rmse_per_sample']=np.sqrt(np.mean(shaped[:,0]**2,1)).tolist();row['coordinate']['rmse_per_physical_time_per_sample']=np.sqrt(np.mean(shaped**2,2)).tolist()
                        bshaped=(baseline_samples-target).reshape(particles,len(grid.times),-1);row['fixed']['initial_rmse_per_sample']=np.sqrt(np.mean(bshaped[:,0]**2,1)).tolist();row['fixed']['rmse_per_physical_time_per_sample']=np.sqrt(np.mean(bshaped**2,2)).tolist()
                        clean=obs.operator.apply(truth).numpy()
                        row['coordinate']['sensor_rmse_against_clean_reference_per_sample']=np.sqrt(np.mean((np.asarray(report['observation_consistency']['predictions_per_sample'])-clean)**2,1)).tolist()
                        row['fixed']['sensor_rmse_against_clean_reference_per_sample']=torch.sqrt(((obs.operator.apply(result.samples[0])-torch.from_numpy(clean))**2).mean(1)).tolist()
                        records.append(row)
                        prefix=f'case{case}-n{count}-sigma{noise}-seed{seed}-q{resolution}'
                        atomic_write(root/(prefix+'.npz'),lambda f:np.savez_compressed(f,coordinate=generated,fixed=baseline_samples,reference=target,queries=q.numpy()))
                    write_json(root/f'case{case}-n{count}-sigma{noise}-seed{seed}-observations.json',{'problem':p.to_dict(),'rows':rows,'observations':obs.to_dict(),'clean_reference_values_evaluation_only':obs.operator.apply(truth).tolist()})
    cp=flex.cp
    base_checkpoint,_,_=_read_checkpoint(baseline)
    baseline_budget={k:base_checkpoint.get(k) for k in ('training_seconds','dataset_sha256','training_ids')}
    del base_checkpoint
    report={'schema_version':1,'kind':'fixed_vs_coordinate_comparison','source':code_identity(),'held_out_cases':case_metadata,'coordinate_checkpoint':str(Path(coordinate).resolve()),'coordinate_sha256':file_digest(coordinate),'fixed_checkpoint':str(Path(baseline).resolve()),'fixed_sha256':file_digest(baseline),'training_budgets':{'coordinate_flow_updates':cp['progress'],'coordinate_autoencoder_updates':cp['config']['ae_steps'],'coordinate_configuration':cp['config'],'coordinate_flow_training_seconds':cp.get('training_seconds'),'coordinate_flow_setup_seconds':cp.get('setup_seconds'),'coordinate_data_cost':cp.get('data_cost'),'fixed_recorded_budget':baseline_budget,'timing_limit':'Missing measurements from older exports are null; never estimated as executed training time','coordinate_ae_dependency':cp['dependency'],'fixed_updates':fixed.metadata['step'],'coordinate_dataset_identity':cp['dataset_identity'],'fixed_training_config':fixed.config.to_dict()},'parameter_counts':{'coordinate_encoder_decoder_velocity':sum(p.numel() for m in [flex.ae,flex.velocity] for p in m.parameters()),'fixed_velocity':sum(p.numel() for p in fixed.model.parameters())},'records':records,'comparison_limits':['Unmatched training data/compute unless user deliberately trains a matched experiment; budgets recorded, no superiority claim','Common reference interpolated away from native nodes; fine-scale accuracy unestablished','Fixed model uses endpoint observation+PDE guidance; coordinate model learned observation conditioning; methods recorded rather than claimed equal algorithms','RSS is process lifetime; CUDA allocated peak only where explicitly recorded; no MPS peak claim','No empirical coverage/calibrated posterior claim; sample errors and ensemble mean reported separately'],'boundary':'Both periodic constructions; coordinate sinusoidal features and periodic baseline interpolation enforce endpoint equality by construction, not learned BC validation'}
    write_json(root/'comparison.json',report)
    lines=['# Fixed-grid versus latent-coordinate heat comparison','','Bounded validation experiment; no scientific superiority or fine-scale truth claim.','',f'Cases {cases}; seeds {seeds}; ensemble {particles}; Heun steps {steps}.','', '| Sensors | sigma | query | coordinate mean sample RMSE | fixed mean sample RMSE |','|---:|---:|---|---:|---:|']
    for row in records:lines.append(f'| {row["spatial_sensors"]} | {row["absolute_sigma"]} | {row["query_shape"]} | {np.mean(row["coordinate"]["rmse_per_sample"]):.6g} | {np.mean(row["fixed"]["rmse_per_sample"]):.6g} |')
    lines+=['',*report['comparison_limits']];(root/'report.md').write_text('\n'.join(lines)+'\n')
    return report
