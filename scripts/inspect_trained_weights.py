"""Read-only actual stage/dependency/optimizer inventory using strict checkpoint loading."""
from pathlib import Path
import json,torch,hashlib
from pdeflow.coordinate.training import read_checkpoint
from pdeflow.coordinate.optimization import scientific_source,options
from pdeflow.coordinate.diagnostic_inventory import inventory
from pdeflow.runtime import file_digest,digest
import argparse
p=argparse.ArgumentParser();p.add_argument('--run-directory',required=True);p.add_argument('--output',required=True);args=p.parse_args()
r=Path(args.run_directory).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
records={};checkpoints={}
for stage,name in [('autoencoder','autoencoder.pt'),('evolution','evolution.pt'),('flow','coordinate.pt')]:
 path=r/stage/name;cp,c=read_checkpoint(path,stage);checkpoints[stage]=cp
 resume=torch.load(r/stage/'resume.pt',weights_only=True,map_location='cpu',mmap=True)
 history=json.loads((r/stage/'history.json').read_text());records[stage]={
 'checkpoint':str(path),'sha256':file_digest(path),'schema':cp['schema'],'engine_version':cp.get('engine_version'),'source_matches_current':cp['scientific_source']['sha256']==scientific_source()['sha256'],
 'progress':cp['progress'],'attempted_updates':cp.get('attempted_updates'),'skipped_updates':cp.get('skipped_updates'),'effective_examples_processed':cp.get('effective_examples_processed'),'completed':cp['completed'],
 'dependency':cp.get('dependency'),'optimizer_state':bool(resume.get('optimizer',{}).get('state')),'scheduler':resume.get('scheduler'),'scaler':resume.get('scaler'),'rng_present':{k:k in resume for k in ('rng','torch_rng','ambient_rng','cuda_rng')},
 'config':c.to_dict(),'dataset_identity':cp['dataset_identity'],'normalizations':cp['normalizations'],'normalization_fitted_ids':cp['normalization_fitted_ids'],'actual_precision':cp.get('actual_precision'),
 'history':{'rows':len(history),'first':{k:history[0].get(k) for k in ('step','training_loss','validation_loss','learning_rate','actual_forward_microbatch')},'last':{k:history[-1].get(k) for k in ('step','training_loss','validation_loss','learning_rate','actual_forward_microbatch')},'validation_rows':[{'step':x['step'],'loss':x['validation_loss'],'policy':x.get('validation_policy')} for x in history if x.get('validation_loss') is not None]},
 'actual_resume_source':resume.get('scientific_source',{}).get('sha256'),'resume_progress':resume['progress'],'selection':'Latest successful optimizer boundary; no best checkpoint file/selection present', 'training_provenance':cp.get('training_provenance')}
 del resume
# Validate dependency hashes before combining any weights.
a,e,b=[checkpoints[s] for s in ('autoencoder','evolution','flow')]
assert e['dependency']['sha256']==records['autoencoder']['sha256']
assert b['dependency']['sha256']==records['evolution']['sha256']
assert a['dataset_identity']==e['dataset_identity']==b['dataset_identity']
assert a['normalizations']==e['normalizations']==b['normalizations']
frozen=[k for k in a['autoencoder'] if k.startswith(('encoder.','decoder.','coefficient.'))]
assert all(torch.equal(a['autoencoder'][k],e['autoencoder'][k]) for k in frozen)
assert all(torch.equal(v,e['autoencoder'][k]) for k,v in b['inference_state'].items())
export=torch.load(r/'export/coordinate.pt',weights_only=True,map_location='cpu',mmap=True)
assert all(torch.equal(v,export['velocity'][k]) for k,v in b['velocity'].items())
for k in ('latent_mean','latent_scale'):assert torch.equal(b[k],export[k])
result={'run_id':json.loads((r/'session.json').read_text())['name'],'current_scientific_source':scientific_source()['sha256'],'stages':records,'lineage_verified':True,'a1_frozen_tensors_equal_in_a2':len(frozen),'a2_tensors_equal_in_b':len(b['inference_state']),'export_flow_and_calibration_equal':True,'flow_inventory':inventory(r/'flow/coordinate.pt')}
(out/'weight-inventory.json').write_text(json.dumps(result,indent=2)+'\n')
print('RUN',result['run_id'],'science',result['current_scientific_source'],'lineage passed')
for s,x in records.items():print(s,x['progress'],x['attempted_updates'],x['skipped_updates'],x['effective_examples_processed'],'source_match',x['source_matches_current'],'LR',x['history']['first']['learning_rate'],x['history']['last']['learning_rate'])
print(result['flow_inventory']['parameter_counts_excluding_buffers'])
