"""Verify immutable downloaded bundle members and safely extract preserved source."""
from pathlib import Path
import hashlib,json,zipfile,stat
import argparse
p=argparse.ArgumentParser();p.add_argument('--artifacts',required=True);p.add_argument('--output',required=True);args=p.parse_args()
root=Path.cwd();inputs=Path(args.artifacts).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()
files={str(p.relative_to(inputs)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs.rglob('*') if p.is_file()}
if (out/'artifact-inventory.json').exists():
 previous=json.loads((out/'artifact-inventory.json').read_text())
 if previous['files']!=files:raise ValueError('Original artifact files changed; preservation baseline is not overwritten')
report={'input_root':str(inputs),'files':files,'bundles':[],'archives':[]}
for folder in sorted(inputs.iterdir()):
 if not folder.is_dir():continue
 manifests=list((folder/'bundles').glob('*.json'))
 for p in manifests:
  x=json.loads(p.read_text());checks=[]
  for n,e in x.get('files',{}).items():
   f=folder/n;checks.append({'name':n,'present':f.is_file(),'matches':f.is_file() and files[str(f.relative_to(inputs))]['sha256']==e['sha256']})
  if not all(x['present'] and x['matches'] for x in checks):raise ValueError('Missing or checksum-mismatched bundle member: '+str(p))
  report['bundles'].append({'folder':folder.name,'manifest':str(p.relative_to(inputs)),'metadata':{k:v for k,v in x.items() if k!='files'},'members':checks})
 for p in folder.rglob('*.zip'):
  target=out/'preserved-source'/folder.name
  with zipfile.ZipFile(p) as z:
   for i in z.infolist():
    n=Path(i.filename)
    assert not n.is_absolute() and '..' not in n.parts and stat.S_IFMT(i.external_attr>>16)!=stat.S_IFLNK
   assert sum(i.file_size for i in z.infolist())<512*1024**2 and z.testzip() is None
   manifest=json.loads(z.read('pdeflow/SOURCE_MANIFEST.json'))
   for name,e in manifest['files'].items():
    data=z.read('pdeflow/'+name);assert len(data)==e['bytes'] and hashlib.sha256(data).hexdigest()==e['sha256']
   assert set(z.namelist())=={'pdeflow/'+n for n in manifest['files']}|{'pdeflow/SOURCE_MANIFEST.json'}
   if target.exists():
    for name,e in manifest['files'].items():assert sha(target/'pdeflow'/name)==e['sha256']
   else:z.extractall(target)
   report['archives'].append({'path':str(p.relative_to(inputs)),'sha256':sha(p),'verified_members':len(manifest['files']),'extracted':str(target.resolve())})
(out/'artifact-inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print('FILES',len(files),'GiB',sum(x['bytes'] for x in files.values())/1024**3)
for b in report['bundles']:
 print(b['folder'],{k:b['metadata'].get(k) for k in ('run_id','kind','available','incomplete_stages','schema')},'members',len(b['members']),'missing',[x['name'] for x in b['members'] if not x['present']],'mismatch',[x['name'] for x in b['members'] if x['present'] and not x['matches']])
print('archives',report['archives'])
