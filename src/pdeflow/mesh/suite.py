"""Explicit individual/suite execution, durable candidate order and no silent restarts."""
from pathlib import Path
from dataclasses import replace
import time,uuid,shutil
from pdeflow.runtime import write_json,read_json,digest,file_digest
from pdeflow.local import run_lock,is_locked
from .contracts import CANDIDATES,preset,MeshConfig
from .session import MeshSession,plan
from .provenance import scientific_source


def suite_plan(capacity='medium',device='cuda',precision='auto',candidates=None,overrides=None,include_legacy=False):
    selected=list(CANDIDATES) if candidates is None else list(candidates)
    if not selected or len(set(selected))!=len(selected) or set(selected)-CANDIDATES.keys():raise ValueError('Choose distinct implemented M1–M5 architectures')
    configs={name:preset(capacity,name,device,precision,overrides).to_dict() for name in selected};legacy=None
    if include_legacy:
        from pdeflow.colab import training_preset
        first=MeshConfig(**next(iter(configs.values()))).validate();ref=first.reference_config();legacy=training_preset(capacity,device,precision,{**first.data,'family_weights':first.family_weights,'time_interval':first.interval,'lengths':tuple(first.domain['lengths']),'origins':tuple(first.domain['origins']),'field_loss':first.field_loss,'seed':first.seed}).to_dict()
    return {'schema':'mesh_suite_v1','order':selected+(['legacy_factorized_v2'] if legacy else []),'configs':configs,'legacy_config':legacy,'legacy_selection':'Explicit optional original A1/A2/B trainer; no checkpoint conversion','science_identity':digest({k:{key:cfg[key] for key in ('data','family_weights','domain','interval','seed')} for k,cfg in configs.items()}),'comparison':'From scratch unless an explicit recorded warm start is selected. Pretraining cost is counted separately; capacity names do not imply parameter matching.','launches_training':False}


class Suite:
    def __init__(self,root,persistent=None):self.root=Path(root).resolve();self.persistent=Path(persistent).resolve() if persistent else None
    @classmethod
    def new(cls,plan,local_root,persistent_root=None):
        identifier=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:10];result=cls(Path(local_root)/identifier,Path(persistent_root)/identifier if persistent_root else None);result.root.mkdir(parents=True);
        from pdeflow.source_bundle import package
        identity=package(result.root/'source/source.zip');identity['scientific_source']=scientific_source();write_json(result.root/'source/identity.json',identity)
        write_json(result.root/'suite.json',{'plan':plan,'state':'planned','candidates':{},'source':scientific_source()});return result
    def inspect(self):return read_json(self.root/'suite.json')
    def sync(self,tag):
        if self.persistent:
            from pdeflow.persistence import publish
            return publish(self.root,self.persistent,keep=3,tag=tag)
    def stop(self):
        write_json(self.root/'stop-request.json',{'time':time.time()});record=self.inspect();active=record.get('active')
        if active:write_json(self.root/'runs'/active/'stop-request.json',{'time':time.time()})
    def preflight(self,updates=8,warmup=2,candidate=None):
        record=self.inspect();reports={}
        for name in ([candidate] if candidate else record['plan']['order']):
            if name=='legacy_factorized_v2':
                from pdeflow.colab import Session
                session=Session(record['plan']['legacy_config'],'legacy_factorized_v2',self.root/'runs',self.root/'unused-store',allow_cpu_check=True,persistence_enabled=False);reports[name]=session.preflight(updates,warmup)
            else:
                session=MeshSession(MeshConfig(**record['plan']['configs'][name]),self.root/'runs'/name);reports[name]=session.preflight(updates,warmup)
        write_json(self.root/'profiles.json',{**(read_json(self.root/'profiles.json') if (self.root/'profiles.json').exists() else {}),**reports});self.sync('preflight');return reports
    def run(self,selected=None):
        with run_lock(self.root):
            record=self.inspect()
            if record['source']['sha256']!=scientific_source()['sha256']:raise ValueError('Suite source changed; reload recorded source')
            if (self.root/'stop-request.json').exists():(self.root/'stop-request.json').unlink()
            self.sync('before-training')
            for name in record['plan']['order']:
                if selected and name not in selected:continue
                previous=record['candidates'].get(name,{})
                if previous.get('state')=='completed':
                    if name!='legacy_factorized_v2':__import__('pdeflow.mesh.artifacts',fromlist=['verify']).verify(self.root/'runs'/name,'resumable')
                    continue
                if (self.root/'stop-request.json').exists():break
                record.update(state='running',active=name);write_json(self.root/'suite.json',record)
                if name=='legacy_factorized_v2':
                    from pdeflow.colab import Session
                    session=Session(record['plan']['legacy_config'],name,self.root/'runs',self.root/'unused-store',allow_cpu_check=True,persistence_enabled=False);outcome=session.run(mode='resume' if (session.root/'session.json').exists() else 'new',override_gate=False);session.finalize()
                else:
                    c=MeshConfig(**record['plan']['configs'][name]).validate();session=MeshSession(c,self.root/'runs'/name)
                    if not (session.root/'source/source.zip').exists():
                        (session.root/'source').mkdir(exist_ok=True);shutil.copy2(self.root/'source/source.zip',session.root/'source/source.zip');shutil.copy2(self.root/'source/identity.json',session.root/'source/identity.json')
                    if max(c.budgets[s] for s in c.kind['stages'])>20:
                        profiles=read_json(self.root/'profiles.json') if (self.root/'profiles.json').exists() else {};profile=profiles.get(name)
                        if not profile or not profile['success'] or profile['requested_config']!=digest(c.to_dict()) or profile['scientific_source']!=scientific_source()['sha256']:raise ValueError('Run preflight for the exact selected candidate/objective before substantial suite training')
                        write_json(session.root/'profile.json',profile)
                    outcome=session.run(event=lambda e:self.sync('candidate-checkpoint'))
                record['candidates'][name]={'state':outcome['state'],'root':str(session.root),'stages':outcome.get('stages')};write_json(self.root/'suite.json',record);self.sync('candidate-transition')
                if outcome['state']!='completed':record.update(state='needs_review' if outcome['state']=='needs_review' else 'interrupted');write_json(self.root/'suite.json',record);return record
            record.update(state='completed' if all(record['candidates'].get(n,{}).get('state')=='completed' for n in record['plan']['order']) else 'partial',active=None);write_json(self.root/'suite.json',record);self.sync('suite-final');return record
    @classmethod
    def from_store(cls,persistent_root,run_id,local_root,snapshot=None):
        from pdeflow import persistence
        if Path(run_id).name!=run_id:raise ValueError('Plain suite ID required')
        store=Path(persistent_root)/run_id;record=persistence.verify(store,snapshot) if snapshot else persistence.latest(store)
        if not record:raise ValueError('No verified suite snapshot')
        root=Path(local_root)/run_id;persistence.restore(store,root,record['id']);result=cls(root,store)
        if result.inspect()['source']['sha256']!=scientific_source()['sha256']:raise ValueError('Reload the source recorded in restored suite')
        return result
