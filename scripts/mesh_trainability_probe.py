"""Bounded fixed-TRAIN-subset diagnostic; no claim of held-out accuracy."""
from pathlib import Path
from dataclasses import replace
import argparse
import torch
from pdeflow.mesh.contracts import preset,CANDIDATES
from pdeflow.mesh.training import train_stage
from pdeflow.mesh.session import MeshSession
from pdeflow.mesh.provenance import counts,scientific_source
from pdeflow.mesh.evaluation import create_cases,compare
from pdeflow.runtime import read_json,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--updates',type=int,default=20);p.add_argument('--device',default='cpu');a=p.parse_args()
 if not 2<=a.updates<=40:raise ValueError('Bounded2..40 diagnostic updates only')
 root=Path(a.output);root.mkdir(parents=True,exist_ok=False);results={};checkpoints={}
 for name in CANDIDATES:
  c=preset('pipeline-check',name,a.device,'fp32',{'budgets':{s:a.updates for s in preset().budgets},'checkpoint_every':a.updates,'validation_every':a.updates,'data':{'query_points':24}});session=MeshSession(c,root/name);record=session.run(fixed_subset=True)
  stages={}
  for stage in c.kind['stages']:
   h=read_json(session.root/'stages'/stage/'history.json');stages[stage]={'first_objective':h[0]['training_loss'],'last_objective':h[-1]['training_loss'],'minimum':min(r['training_loss'] for r in h),'decreased':h[-1]['training_loss']<h[0]['training_loss'],'updates':len(h),'family_counts':h[0]['family_example_counts'],'components':h[-1]['components'],'validation':read_json(session.root/'stages'/stage/'validation.json')}
  results[name]={'stages':stages,'state':record['state'],'parameters':counts(c)};checkpoints[name]=str(session.root/'export/model.pt');write_json(root/'trainability.json',{'candidates':results,'device':a.device,'scientific_source':scientific_source(),'claim':'Fixed TRAIN cases/noise/time learning only; not held-out accuracy or posterior calibration'});print(name,stages,flush=True)
 create_cases(c,root/'cases',count=2);compare(checkpoints,root/'cases',root/'comparison',a.device,particles=2,steps=4)

if __name__=='__main__':main()
