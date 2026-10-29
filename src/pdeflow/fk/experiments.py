"""Bounded learned-prior comparison and explicit CPU/MPS execution probes."""
from dataclasses import replace,asdict
from pathlib import Path
import json,time,math
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,read_json,write_json,environment,code_identity
from pdeflow.training import load_inference
from pdeflow.core import ProblemSpec
from pdeflow.observations import ObservationSet
from pdeflow.sampling import sample
from pdeflow.evaluation import export_result,metrics


def inputs(path):
 p=ProblemSpec.from_dict(read_json(path/'problem.json'));o=ObservationSet.from_dict(read_json(path/'observations.json'),p.grid)
 return p,o


def comparison(output,checkpoint,evaluation,device='cpu',particles=16,steps=8,seeds=(11,23,37)):
 root=Path(output)
 if root.exists():raise FileExistsError('Choose a fresh comparison output')
 root.mkdir(parents=True);runtime=RuntimeSpec(device=device);b=load_inference(checkpoint,runtime)
 evaluation=Path(evaluation);val,vo=inputs(evaluation/'validation/0');p,o=inputs(evaluation/'test/0')
 base=replace(b.config.inference,particles=particles,steps=steps,controller={'max_particles':max(16,particles),'max_network_evaluations':max(256,steps*(math.ceil(particles/min(8,particles))+4)),'max_sampling_steps':steps},fk={'chunk_size':min(8,particles),'max_particle_evaluations':max(4096,particles*steps),'ess_fraction':.7,'resample_every':2})
 tuning=[]
 for obs_weight in [.0001,.001]:
  for physics_weight in [0.,.0001]:
   inf=replace(base,seed=61,fk={**base.fk,'observation_weight':obs_weight,'physics_weight':physics_weight})
   r=sample(b,val,vo,inf,'fk');m=metrics(r,val,vo)
   tuning.append({'observation_weight':obs_weight,'physics_weight':physics_weight,'validation_score':m['observation_rmse']+.001*m['pde_fd_rmse']})
 selected=min(tuning,key=lambda x:x['validation_score'])
 gradient=[]
 for observation_strength in [.1,.3]:
  inf=replace(base,seed=61,observation_strength=observation_strength,physics_strength=.0001)
  r=sample(b,val,vo,inf,'observation_pde');m=metrics(r,val,vo)
  gradient.append({'observation_strength':observation_strength,'physics_strength':.0001,'validation_score':m['observation_rmse']+.001*m['pde_fd_rmse']})
 guided=min(gradient,key=lambda x:x['validation_score']);inf=replace(base,fk={**base.fk,'observation_weight':selected['observation_weight'],'physics_weight':selected['physics_weight']},observation_strength=guided['observation_strength'],physics_strength=guided['physics_strength'])
 write_json(root/'config.json',replace(b.config,runtime=runtime,inference=inf).to_dict())
 write_json(root/'validation-selection.json',{'candidate_FK':tuning,'gradient_candidates':gradient,'selected':selected,'selected_gradient':guided,'criterion':'validation physical observation RMSE + .001*FD residual; reference-free validation only; no test-driven selection','fixed_noise':'.2 strength vanishing noise schedule; ESS .7, eligible every2 chosen before PDE tests from analytic verification'})
 # Reference is opened only after all inference and validation selection.
 results=[]
 for seed in seeds:
  for method in ['unguided','observation_pde','importance','fk']:
   r=sample(b,p,o,replace(inf,seed=seed),method);results.append((seed,method,r))
 with np.load(evaluation/'test/0/reference.npz',allow_pickle=False) as data:truth=torch.from_numpy(data['values'].copy())
 records=[]
 for seed,method,r in results:
  m=export_result(root/f'seed-{seed}'/method,r,p,o,truth)
  records.append({'seed':seed,'method':method,'observation_rmse':m['observation_rmse'],'pde_fd_rmse':m['pde_fd_rmse'],'pde_spectral_space_rmse':m['pde_spectral_space_rmse'],'unobserved_relative_l2':m['relative_l2_unobserved_mean'],'spread':m['ensemble_rms_spread'],'seconds':r.diagnostics['seconds'],'model_calls':r.diagnostics['model_evaluations'],'model_particle_evaluations':r.diagnostics.get('model_particle_evaluations',r.diagnostics['model_evaluations']*particles),'gradient_evaluations':r.diagnostics['gradient_evaluations'],'energy_particle_evaluations':r.diagnostics.get('energy_particle_evaluations'),'ess':r.diagnostics.get('fk',{}).get('weight_summary'),'unique_ancestors':r.diagnostics.get('fk',{}).get('unique_ancestors'),'stop_reason':r.stop_reason,'terminal_complete':r.provenance.get('terminal_target_complete',True)})
 summary={method:{k:float(np.mean([r[k] for r in records if r['method']==method])) for k in ['observation_rmse','pde_fd_rmse','unobserved_relative_l2','spread','seconds','model_calls','model_particle_evaluations']} for method in ['unguided','observation_pde','importance','fk']}
 report={'inference_code':code_identity(),'checkpoint':str(checkpoint),'checkpoint_identity':b.metadata['checkpoint_sha256'],'evaluation':str(evaluation),'seeds':list(seeds),'particles':particles,'steps':steps,'records':records,'summary':summary,'hardware':environment(runtime),'scientific_claim':False,'note':'One fixed held-out problem, three sampling seeds, undertrained checkpoint. Different measured computational costs. FK/IS share the same numerical SDE base and terminal energy; gradient heuristic and ODE baseline have different laws. No analytic posterior under a different prior is used.'}
 write_json(root/'comparison.json',report)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(7,4))
 for method,row in summary.items():ax.scatter(row['model_particle_evaluations'],row['unobserved_relative_l2'],label=method)
 ax.set(xlabel='Particle model evaluations (gradient costs additional)',ylabel='Unobserved relative L2',title='Heat smoke: quality and actual inference cost; no winning claim');ax.legend();fig.tight_layout();fig.savefig(root/'quality-cost.png');plt.close(fig)
 return report


def backend_check(output,device='mps'):
 root=Path(output)
 if root.exists():raise FileExistsError('Choose a fresh backend check output')
 runtime=RuntimeSpec(device=device);runtime.resolve();root.mkdir(parents=True)
 reports={};cases=[('heat2','outputs/accepted_cpu_2d','data/test_problem.json','data/test_observations.json'),('heat3','outputs/accepted_cpu_3d','data/test_problem.json','data/test_observations.json'),('darcy2','outputs/multispecialist-20261021/cpu-2d','data/test_problem.json','data/test_observations.json'),('chemistry3','outputs/online-chemistry-20261021/chemistry-3d','evaluation/test/0/problem.json','evaluation/test/0/observations.json')]
 for name,path,pp,op in cases:
  path=Path(path);b=load_inference(path/'training/inference.pt',runtime);p=ProblemSpec.from_dict(read_json(path/pp));o=ObservationSet.from_dict(read_json(path/op),p.grid)
  inf=replace(b.config.inference,particles=8,steps=4,controller={'max_particles':8,'check_interval':1},fk={'chunk_size':4,'ess_fraction':1.,'resample_every':1,'observation_convention':'tempered_mean','observation_weight':.01,'physics_weight':.0001})
  r=sample(b,p,o,inf,'fk');assert r.provenance['terminal_target_complete'] and any(e['action']=='RESAMPLE' for e in r.diagnostics['process']['actions'])
  export_result(root/name,r,p,o)
  cpu=sample(load_inference(path/'training/inference.pt',RuntimeSpec()),p,o,inf,'fk')
  from pdeflow.evaluation.weighted import mean,weights
  difference=float((mean(r.samples[0].double(),weights(r))-mean(cpu.samples[0].double(),weights(cpu))).square().mean().sqrt())
  with np.load(root/name/'result.npz') as a:assert np.isfinite(a['samples']).all() and np.allclose(a['weights'].sum(1),1)
  reports[name]={'actual_model_device':str(next(b.model.parameters()).device),'terminal_complete':True,'resampling_events':r.diagnostics['resampling_events'],'cpu_reload':'NPZ weights/fields read and CPU checkpoint sampled','cpu_mps_weighted_mean_rmse':difference,'cost':{k:r.diagnostics[k] for k in ['model_evaluations','model_particle_evaluations','energy_particle_evaluations','seconds','weight_transfer_seconds']}}
  if name=='heat2':
   guided=sample(b,p,o,replace(inf,fk=None),'observation_pde');export_result(root/'gradient-controller',guided,p,o)
   fine=load_inference('outputs/process-20261025/heat-fine-smoke/training/inference.pt',runtime)
   refine=replace(inf,fk=None,controller={'max_attempts':2,'refine':True,'check_interval':1,'max_sampling_steps':8})
   refined=sample(b,p,o,refine,'observation_pde',fine_bundle=fine)
   q=ProblemSpec.from_dict(refined.provenance['final_problem']);qo=ObservationSet.from_dict(refined.provenance['final_observations'],q.grid)
   assert q.grid.shape==(16,16);export_result(root/'gradient-refinement',refined,q,qo)
   cancel=sample(b,p,o,inf,'fk',cancelled=lambda:True);assert cancel.stop_reason=='STOP_CANCELLED' and not cancel.provenance['terminal_target_complete'];export_result(root/'cancelled',cancel,p,o)
   reports['gradient_controller']='executed with physical/observation final report';reports['gradient_refinement']='actual MPS8²→16²; FK plus refinement rejected';reports['cancellation']='STOP_CANCELLED, incomplete path, exported state; no producer workers'
 write_json(root/'backend.json',{'hardware':environment(runtime),'reports':reports,'cuda':'untested','actual_colab':'untested'})
 return reports


def main():
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['comparison','backend']);parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--checkpoint',default='outputs/accepted_cpu_2d/training/inference.pt');parser.add_argument('--evaluation',default='outputs/online-chemistry-20261021/throughput-final-heat/evaluation');parser.add_argument('--particles',type=int,default=16);parser.add_argument('--steps',type=int,default=8);a=parser.parse_args()
 result=backend_check(a.output,a.device) if a.action=='backend' else comparison(a.output,a.checkpoint,a.evaluation,a.device,a.particles,a.steps)
 print(result.get('summary',result))
if __name__=='__main__':main()
