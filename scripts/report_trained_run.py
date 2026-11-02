"""Summarize already-executed trained-model diagnostics; never runs a model."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);args=p.parse_args();root=Path(args.analysis).resolve();figures=root/'figures';figures.mkdir(exist_ok=True)
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
inv=read(root/'weight-inventory.json');updates_label='/'.join(str(inv['stages'][s]['progress']) for s in ('autoencoder','evolution','flow'));oracles=read(root/'oracles/metrics.json');summary={'run_id':inv['run_id'],'checkpoint_sha256':{k:x['sha256'] for k,x in inv['stages'].items()},'oracle_aggregates':{},'conditional_cases':{},'correction':{},'aggregation':'Equal validation examples WITHIN each family; each quantity separate. Aggregate RMSE=sqrt(mean case MSE), never mean case RMSE. Spatial periodic quadrature and explicit trapezoid physical-time integration. No pooling of u/v physical units.'}
for family in ('heat','wave'):
 records=[r for r in oracles['records'] if r['family']==family]
 for q in ('u',) if family=='heat' else ('u','v'):
  key=f'{family}/{q}';row={'examples':len(records)}
  for name in ('a1_initial','a2','held_initial'):
   mse=np.mean([r['quantities'][q][name]['ensemble_mean']['uniform_physical']['mse'] for r in records]);row[name]={'mse':float(mse),'rmse':float(np.sqrt(mse))}
  base=np.mean([r['quantities'][q]['a1_initial']['ensemble_mean']['constant_training_mean']['mse'] for r in records]);row['constant_initial_baseline_rmse']=float(np.sqrt(base));row['a2_mse_ratio_to_held_initial']=row['a2']['mse']/row['held_initial']['mse'];row['per_time']=[]
  for t in oracles['times']:
   a=[next(x for x in r['oracle']['per_time'] if x['quantity']==q and x['physical_time']==t) for r in records]
   row['per_time'].append({'time':t,**{n+'_rmse':float(np.sqrt(np.mean([x[n]['mse'] for x in a]))) for n in ('learned','held_initial','reencoded_reference')}})
  features=[]
  for r in records:
   frame=r['quantities'][q]['a1_initial']['ensemble_mean']['frames'][0];regions=frame['regions'];small=[x['mse'] for x in regions if x['area_fraction']<=.05 and x['mse'] is not None];large=[x['mse'] for x in regions if x['area_fraction']>.05 and x['mse'] is not None]
   features.append({'id':r['id'],'regions':regions,'mean_region_mse':frame['mean_region_mse'],'worst_region_mse':frame['worst_region_mse'],'background_mse':frame['background_mse'],'small_region_mean_mse':float(np.mean(small)) if small else None,'large_region_mean_mse':float(np.mean(large)) if large else None})
  row['initial_feature_errors_by_case']=features;summary['oracle_aggregates'][key]=row
  fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
  for metric,label in [('learned_rmse','A2 from encoded initial state'),('held_initial_rmse','Same initial latent held constant'),('reencoded_reference_rmse','Reference-state AE diagnostic (not error floor)')]:ax.plot([x['time'] for x in row['per_time']],[x[metric] for x in row['per_time']],'o-',label=label)
  ax.set(xlabel='Known physical time',ylabel=f'Physical {q} RMSE',title=f'{family} {q} · run {inv["run_id"]}\nA2 {inv["stages"]["evolution"]["sha256"][:12]} · 2500 updates · {len(records)} validation cases');ax.legend(fontsize=8);fig.savefig(figures/f'oracle-{family}-{q}.png',dpi=150);plt.close(fig)
for folder in sorted((root/'reconstruction').iterdir()):
 if not (folder/'diagnosis.json').exists():continue
 report=read(folder/'diagnosis.json');case=read(root/'cases'/folder.name/'case.json');r={'family':case['problem']['family'],'physical_times':case['problem']['grid']['times'],'quantity_metrics':{},'observation_distribution':report['conditioning_distribution'],'reference_assessment':report['distribution_assessment'],'parameters':report['stages']['b']['production_report']['inferred_parameters'],'true_parameter_evaluation_only':case['true_parameter_evaluation_only']}
 for q in report['stages']['b']['quantities']:
  values={}
  for stage in ('a1','a2','b'):
   x=report['stages'][stage];values[stage]={'initial_rmse':x['initial_state_metrics'][q]['ensemble_mean']['uniform_physical']['rmse']}
   if stage!='a1':
    sensor=x['conditioning_sensor_metrics']['by_quantity'].get(q);held=x['heldout_sensor_metrics']['by_quantity'].get(q)
    values[stage].update(trajectory=x['quantities'][q]['ensemble_mean']['uniform_physical'],spread_rms=x['quantities'][q]['spread_rms'],per_sample_trajectory_rmse=[m['uniform_physical']['rmse'] for m in x['quantities'][q]['per_sample']],conditioning=sensor,heldout=held)
  r['quantity_metrics'][q]=values
 summary['conditional_cases'][folder.name]=r
 with np.load(folder/'fields.npz',allow_pickle=False) as fields:
  for q in r['quantity_metrics']:
   reference=fields['reference_'+q];a1=fields['a1_initial_'+q][0,0];a2=fields['a2_'+q][0];b=fields['b_'+q].mean(0);times=np.asarray(case['problem']['grid']['times']);selected=sorted(set(int(np.argmin(abs(times-t))) for t in (0.,.05,.237,1.)))
   arrays=[reference,np.repeat(a1[None],len(times),0),a2,b];lo=min(float(x.min()) for x in arrays);hi=max(float(x.max()) for x in arrays);errors=[abs(a2-reference),abs(b-reference)];error_max=max(float(x.max()) for x in errors)
   fig,axes=plt.subplots(len(selected),6,figsize=(18,3*len(selected)),layout='constrained');names=['Reference','A1 initial held constant','A2 true-initial oracle','B ensemble mean','Absolute A2 error','Absolute B error']
   L=case['problem']['grid']['domain']['lengths'];o=case['problem']['grid']['domain']['origins'];dx=L[0]/reference.shape[-2];dy=L[1]/reference.shape[-1]
   for row,k in enumerate(selected):
    for col,a in enumerate([*arrays,*errors]):
     displayed=np.pad(a[k],((0,1),(0,1)),mode='wrap');im=axes[row,col].imshow(displayed.T,origin='lower',extent=[o[0]-dx/2,o[0]+L[0]+dx/2,o[1]-dy/2,o[1]+L[1]+dy/2],vmin=lo if col<4 else 0,vmax=hi if col<4 else error_max,cmap='RdBu_r' if col<4 else 'magma');axes[row,col].set_xlim(o[0],o[0]+L[0]);axes[row,col].set_ylim(o[1],o[1]+L[1]);axes[row,col].set_title(f'{names[col]}\n{r["family"]} {q}, t={times[k]:.4g}',fontsize=9)
     if col==0:field_im=im
   fig.colorbar(field_im,ax=axes[:,:4].ravel().tolist(),label=f'Physical {q}; common reference/prediction scale');fig.colorbar(im,ax=axes[:,4:].ravel().tolist(),label='Absolute physical error; common scale')
   fig.suptitle(f'Run {inv["run_id"]} · B {summary["checkpoint_sha256"]["flow"][:12]} · A2 {summary["checkpoint_sha256"]["evolution"][:12]} · A1 {summary["checkpoint_sha256"]["autoencoder"][:12]} · stage updates {updates_label}\n{folder.name} · A1 column is an initial-state baseline, not trajectory encoding. B: all4 samples averaged, no best-sample selection.',fontsize=11);fig.savefig(figures/f'{folder.name}-{q}.png',dpi=120);plt.close(fig)
for folder in (root/'corrections').iterdir():
 if not (folder/'correction.json').exists():continue
 r=read(folder/'correction.json');row={'seconds':r['seconds'],'frozen_weights_unchanged':r['frozen_state_sha256_before']==r['frozen_state_sha256_after'],'steps':r['steps'],'reference_used_in_optimization':r['reference_used_in_optimization'],'before':{},'after':{}}
 for when in ('before','after'):
  x=r[when];row[when]={'sensor_rmse':x['conditioning_sensor_metrics']['ensemble_mean']['rmse'],'heldout_rmse':x['heldout_sensor_metrics']['ensemble_mean']['rmse'],'quantities':{k:v['ensemble_mean']['uniform_physical'] for k,v in x['quantities'].items()},'parameters':x['parameters'],'physics_invariants':x['physics_invariants']}
 with np.load(folder/'fields.npz',allow_pickle=False) as a:
  difference=a['corrected_standardized_latent']-a['initial_standardized_latent'];row['latent_displacement_standardized_rms']=float(np.sqrt(np.mean(difference**2)));row['latent_displacement_reduction']='Full stored latent tensor RMS; inactive known/absent entries are zero; per-sample L2 also retained';row['latent_displacement_l2_per_sample']=np.sqrt(np.sum(difference**2,axis=(1,2))).tolist()
  reference={k[len('reference_'):]:a[k] for k in a.files if k.startswith('reference_')}
  for q,truth in reference.items():
   for when,key in [('before','uncorrected_'+q),('after','corrected_'+q)]:row[when]['quantities'][q]['initial_rmse']=float(np.sqrt(np.mean((a[key].mean(0)[0]-truth[0])**2)))
 summary['correction'][folder.name]=row
history=read(Path(inv['stages']['flow']['checkpoint']).parent/'history.json');d=np.array([r['components']['flow_matching']-r['training_loss'] for r in history]);summary['confirmed_telemetry_defect']={'rows':len(d),'rows_with_abs_difference_above_1e_6':int(np.sum(abs(d)>1e-6)),'max_absolute_difference':float(max(abs(d))),'mean_absolute_difference':float(np.mean(abs(d))),'cause':'Saved flow_matching component was last forward microbatch mean, training_loss was actual full effective batch mean. Actual run beta0 and legacy auxiliary weights0, so true full-batch FM equals training_loss. Original history is immutable.','training_objective_or_weights_changed_by_fix':False}
from dataclasses import asdict
from pdeflow.field_losses import settings
summary['feature_evaluation_policy']={'settings':asdict(settings(inv['stages']['autoencoder']['config']['field_loss'])),'meaning':'Fixed reference-only evaluation segmentation; not the actual training objective (ordinary MSE). Small-region summaries use area_fraction<=0.05, with per-case regions retained. No test-based threshold tuning.'}
write(root/'summary.json',summary)
print(json.dumps(summary['oracle_aggregates'],default=str)[:200]);print('Figures',len(list(figures.glob('*.png'))))
