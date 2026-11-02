"""Read-only collection of one mesh run/store. No restore, initialization or training.

Run in a separate process/fresh kernel. Output is a NEW compact ZIP; large model
objects are health-checked and identified, not copied. Requires stdlib and torch.
"""
from pathlib import Path
import argparse,json,hashlib,time,zipfile,os,socket
import torch


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def load(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
def health(path):
 cp=torch.load(path,map_location='cpu',weights_only=True);bad=[]
 def walk(value,key):
  if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
   if not torch.isfinite(value).all():bad.append(key)
  elif isinstance(value,float) and not __import__('math').isfinite(value):bad.append(key)
  elif isinstance(value,dict):
   for k,v in value.items():walk(v,key+'.'+str(k))
  elif isinstance(value,(tuple,list)):
   for i,v in enumerate(value):walk(v,key+f'[{i}]')
 walk(cp.get('model',{}),'model');walk(cp.get('optimizer',{}),'optimizer')
 ns=cp.get('normalizations',{});normalization_ok=all(__import__('math').isfinite(n.get('mean',0)) and __import__('math').isfinite(n['scale']) and n['scale']>0 for rows in ns.values() for n in rows)
 return {'healthy':not bad and normalization_ok,'invalid_tensor_paths':bad[:30],'successful_updates':cp.get('successful_updates'),'attempted_updates':cp.get('attempted_updates'),'skipped_updates':cp.get('skipped_updates'),'failed_attempts':cp.get('failed_attempts',0),'environment':cp.get('environment'),'precision':cp.get('precision'),'scaler':cp.get('scaler'),'scheduler':cp.get('scheduler'),'config':cp.get('config'),'source':cp.get('scientific_source'),'generator_state_saved':'generator' in cp,'RNG_saved':{k:k in cp for k in ('ambient_rng','torch_rng','cuda_rng')},'recent_history':cp.get('history',[])[-30:]}


def collect(local_run,store,output):
 local=Path(local_run) if local_run else None;store=Path(store) if store else None;output=Path(output)
 if output.exists():raise FileExistsError('Preserve collection outputs; choose a new ZIP filename')
 record={'local_run':str(local),'store':str(store),'collected_unix':time.time(),'snapshots':[],'checkpoints':[],'local_owner':'unavailable','exact_failure_batch_available':False};files={}
 if local and local.exists():
  lock=local/'.run.lock'
  if lock.exists():
   import fcntl
   with lock.open('r') as f:
    try:fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB);record['local_owner']='no local training-owner lock';fcntl.flock(f,fcntl.LOCK_UN)
    except BlockingIOError:record['local_owner']='ACTIVE: read-only snapshot inspection only; do not install/change this source'
  for name in ('session.json','resolved-config.json','source/identity.json','progress.json','sync-status.json','profile.json','stages/reconstruction/run.json','stages/reconstruction/error.json','stages/reconstruction/history.json','data/dataset.json'):
   p=local/name
   if p.is_file() and p.stat().st_size<8*1024**2:files['local/'+name]=p.read_bytes()
  journal=local/'stages/reconstruction/history-live.jsonl'
  if journal.exists():
   with journal.open('rb') as f:f.seek(max(0,journal.stat().st_size-128*1024));files['local/recent-history.jsonl']=f.read()
  for p in (local/'stages/reconstruction/failure').rglob('*'):
   if p.is_file() and (p.suffix=='.json' or p.name=='batch.pt') and p.stat().st_size<8*1024**2:files['local/'+str(p.relative_to(local))]=p.read_bytes();record['exact_failure_batch_available']|=p.name=='batch.pt'
  for p in (local/'stages/reconstruction').glob('**/*.pt'):
   if p.name not in ('resume.pt','previous.pt'):continue
   record['checkpoints'].append({'origin':'local','path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,**health(p)})
 if store and store.exists():
  for manifest in sorted((store/'snapshots').glob('*.json')):
   try:
    r=load(manifest);assert manifest.with_suffix('.commit').read_text()==digest(r)
    for name,e in r['files'].items():
     assert not Path(name).is_absolute() and '..' not in Path(name).parts
     obj=store/'objects'/e['sha256'];assert len(e['sha256'])==64 and all(c in '0123456789abcdef' for c in e['sha256']) and obj.stat().st_size==e['bytes'] and sha(obj)==e['sha256']
    record['snapshots'].append({'id':r['id'],'sequence':r.get('sequence'),'successful_update':r.get('successful_update'),'tag':r.get('tag'),'verified':True})
    for name,e in r['files'].items():
     obj=store/'objects'/e['sha256']
     if name.endswith(('reconstruction/resume.pt','healthy/previous.pt')):record['checkpoints'].append({'origin':'verified Drive snapshot','snapshot':r['id'],'path':str(obj),'sha256':e['sha256'],'bytes':e['bytes'],**health(obj)})
    # Small exact metadata from latest sequence only; never the full unrelated tree.
    if r.get('sequence',0)>=record.get('selected_sequence',-1):
     record['selected_sequence']=r.get('sequence',0)
     for name,e in r['files'].items():
      if name in ('session.json','resolved-config.json','data/dataset.json','source/identity.json','source/source.zip','stages/reconstruction/run.json','stages/reconstruction/error.json') and e['bytes']<8*1024**2:files['latest-snapshot/'+name]=(store/'objects'/e['sha256']).read_bytes()
   except Exception as error:record['snapshots'].append({'manifest':str(manifest),'verified':False,'error':str(error)})
 healthy=[r for r in record['checkpoints'] if r['healthy']];record['highest_healthy_observed']=max(healthy,key=lambda r:r.get('successful_updates',-1)) if healthy else None
 verified=[r for r in healthy if r['origin']=='verified Drive snapshot'];record['highest_healthy_verified_snapshot']=max(verified,key=lambda r:r.get('successful_updates',-1)) if verified else None
 record['replay_limit']='Old source saved no offending batch/pre-attempt RNG on failure; deterministic reconstruction data at successful index can be regenerated with original source/config, but actual CUDA behavior and unsaved history must not be invented'
 output.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
  for name,data in files.items():z.writestr(name,data)
  z.writestr('collection.json',json.dumps(record,indent=2,allow_nan=False))
 print(json.dumps({'output':str(output),'bytes':output.stat().st_size,'healthy_checkpoint':record['highest_healthy_observed'] and {k:record['highest_healthy_observed'][k] for k in ('path','successful_updates','sha256','origin')},'local_owner':record['local_owner']},indent=2));return record

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--local-run');p.add_argument('--store');p.add_argument('--output',required=True);a=p.parse_args();collect(a.local_run,a.store,a.output)
