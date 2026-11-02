"""Local/Colab lifecycle for real mesh candidates; thin notebooks share this code."""
from pathlib import Path
from dataclasses import replace
from contextlib import nullcontext
import time,uuid,shutil,traceback
from pdeflow.runtime import read_json,write_json,file_digest,digest
from pdeflow.local import run_lock,is_locked
from .contracts import MeshConfig,CANDIDATES,model_contract
from .provenance import scientific_source,read_checkpoint,counts
from .training import train_stage
from . import artifacts


def plan(c):
    c=c if isinstance(c,MeshConfig) else MeshConfig(**c).validate()
    return {'architecture':c.architecture,'label':c.kind['name'],'stages':list(c.kind['stages']),'budgets':{s:c.budgets[s] for s in c.kind['stages']},'capacity':c.capacity,'parameters':counts(c),'family_weights':{k:v/sum(c.family_weights.values()) for k,v in c.family_weights.items()},'config':c.to_dict(),'scope':'Initial only' if not c.kind['trajectory'] else 'Complete initial state and learned continuous evolution','batch':{'effective':c.batch_size,'microbatch':c.microbatch,'passes':__import__('math').ceil(c.batch_size/c.microbatch)},'cost_policy':'Pretraining and conditional stages are separate costs. Equal update counts across different objectives are not equal compute.','quality':'No trained/accurate default checkpoint is supplied'}


class MeshSession:
    def __init__(self,c,root,persistent=None,save_updates=100,keep_snapshots=3):
        self.c=(c if isinstance(c,MeshConfig) else MeshConfig(**c)).validate();self.root=Path(root).expanduser().resolve();self.persistent=Path(persistent).expanduser().resolve() if persistent else None;self.save_updates=save_updates;self.keep_snapshots=keep_snapshots
        if type(save_updates)!=int or save_updates<1 or not 1<=keep_snapshots<=20:raise ValueError('Invalid persistence cadence/retention')
        if self.persistent and (self.root==self.persistent or self.root.is_relative_to(self.persistent) or self.persistent.is_relative_to(self.root)):raise ValueError('Local and persistent roots must be separate')
        self.root.mkdir(parents=True,exist_ok=True)
    @classmethod
    def new(cls,c,name,local_root,persistent_root=None,**kwargs):
        identifier=time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'-'+uuid.uuid4().hex[:12]
        result=cls(c,Path(local_root)/identifier,Path(persistent_root)/identifier if persistent_root else None,**kwargs);write_json(result.root/'display-name.json',{'name':name});return result
    @classmethod
    def reopen(cls,root,persistent=None,device=None,**kwargs):
        root=Path(root);record=read_json(root/'session.json');c=MeshConfig(**record['config']).validate()
        if record['scientific_source']['sha256']!=scientific_source()['sha256']:raise ValueError('Reload this run original source ZIP before resuming; no automatic source override')
        if device:c=replace(c,runtime={**c.runtime,'device':device})
        return cls(c,root,persistent,**kwargs)
    @classmethod
    def from_store(cls,persistent_root,run_id,local_root,snapshot=None,device=None,**kwargs):
        from pdeflow import persistence
        if Path(run_id).name!=run_id or '..' in run_id:raise ValueError('Plain run ID required')
        store=Path(persistent_root)/run_id;record=persistence.verify(store,snapshot) if snapshot else persistence.latest(store)
        if not record:raise ValueError('No complete verified snapshot')
        root=Path(local_root)/run_id;persistence.restore(store,root,record['id']);result=cls.reopen(root,store,device,**kwargs)
        if (root/'bundles/resumable.json').exists():artifacts.verify(root,'resumable')
        return result
    def inspect(self):return plan(self.c)
    def stop(self):write_json(self.root/'stop-request.json',{'time':time.time(),'policy':'Stop after current optimizer boundary and save complete state'})
    def sync(self,tag=None):
        if self.persistent:
            from pdeflow.persistence import publish
            return publish(self.root,self.persistent,keep=self.keep_snapshots,tag=tag)
        return None
    def _source(self):
        source=self.root/'source/source.zip';current=scientific_source()
        if source.exists():
            identity=read_json(source.with_name('identity.json'))
            if identity['scientific_source']['sha256']!=current['sha256'] or file_digest(source)!=identity['sha256']:raise ValueError('Recorded scientific source differs; never overwriting it')
        else:
            from pdeflow.source_bundle import package
            identity=package(source);identity['scientific_source']=current;write_json(source.with_name('identity.json'),identity)
    def run(self,stop_after=None,owned=False,event=None,extend=False,fixed_subset=False):
        with nullcontext() if owned else run_lock(self.root):
            manifest=self.root/'session.json';old=read_json(manifest) if manifest.exists() else {}
            if old and model_contract(MeshConfig(**old['config']).validate())!=model_contract(self.c):raise ValueError('Incompatible session configuration; use a NEW experiment')
            if old and old['config']['budgets']!=self.c.budgets and not extend:raise ValueError('Changing budgets requires explicit session.extend(stage, additional)')
            if old and old['scientific_source']['sha256']!=scientific_source()['sha256']:raise ValueError('Changed mesh source; reload the saved source before resume')
            if max(self.c.budgets[s] for s in self.c.kind['stages'])>20:
                profile=read_json(self.root/'profile.json') if (self.root/'profile.json').exists() else None
                if not profile or not profile['success'] or profile['requested_config']!=digest(self.c.to_dict()) or profile['scientific_source']!=scientific_source()['sha256']:raise ValueError('Run preflight for this exact candidate/objective/configuration before substantial training')
            record={**old,'schema':'mesh_session_v1','config':self.c.to_dict(),'initial_config':old.get('initial_config',self.c.to_dict()),'scientific_source':scientific_source(),'state':'preparing','progress':0,'pid':__import__('os').getpid(),'save_updates':self.save_updates,'keep_snapshots':self.keep_snapshots,'persistent':str(self.persistent) if self.persistent else None,'stages':old.get('stages',{}),'source_policy':'Strict recorded source/config; no AE stage exists'}
            (self.root/'stop-request.json').unlink(missing_ok=True);self._source();write_json(manifest,record);write_json(self.root/'resolved-config.json',self.c.to_dict());self.sync('before-training')
            def checkpoint(e):
                stage=e['stage'];record.update(stage=stage,progress=e['record']['progress'],state=e['record']['state']);record['stages'][stage]={'successful_updates':e['record']['progress'],'completed':e['record']['state']=='completed'};write_json(manifest,record)
                if event:event(e)
                if e['record']['progress']%self.save_updates==0 or e['record']['state']!='interrupted':self.sync('checkpoint')
            dependency=None
            try:
                for stage in self.c.kind['stages']:
                    record.update(stage=stage,state='running');write_json(manifest,record);path=self.root/'stages'/stage/'resume.pt';cp=None
                    if path.exists():cp,prior=read_checkpoint(path)
                    complete=bool(cp and cp['completed'] and cp['successful_updates']==self.c.budgets[stage])
                    if not complete:
                        result=train_stage(self.c,self.root,stage,dependency,resume=path.exists(),stop_after=stop_after,should_stop=lambda:(self.root/'stop-request.json').exists(),event=checkpoint,extend=extend,fixed_subset=fixed_subset,checkpoint_policy=lambda s,k:k==1 or k%self.save_updates==0)
                        if result['state']!='completed':record.update(state='interrupted');write_json(manifest,record);self.finalize();return record
                    cp,_=read_checkpoint(path);dependency=path;quality=cp['validation'];record['stages'][stage]={'successful_updates':cp['successful_updates'],'completed':True,'validation':quality,'checkpoint_sha256':file_digest(path)};write_json(manifest,record);self.sync('stage-complete')
                    if not self.c.pipeline_check and quality and not quality['passed']:
                        record.update(state='needs_review',gate={'stage':stage,'ratio':quality['mse_ratio'],'meaning':'Inspect/improve this candidate before later stages/suite; no automatic bypass'});write_json(manifest,record);self.finalize();return record
                record.update(state='completed');write_json(manifest,record);self.finalize();return record
            except BaseException as error:
                record.update(state='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error),traceback=traceback.format_exc());write_json(manifest,record)
                try:self.sync('failed-safe-boundary')
                except Exception as save_error:print('Persistent save also failed:',save_error,flush=True)
                raise
    def finalize(self):
        if not (self.root/'session.json').exists():raise ValueError('No started run to finalize')
        export=artifacts.export(self.root);views=artifacts.build(self.root) if (self.root/'data/dataset.json').exists() else {};snapshot=self.sync('final')
        return {'root':str(self.root),'export':str(export) if export else None,'bundles':{k:{'available':v['available'],'files':len(v['files'])} for k,v in views.items()},'persistent_snapshot':snapshot['id'] if snapshot else None,'verified_local':bool(views),'accuracy':'Unestablished'}
    def preflight(self,updates=8,warmup=2):
        if is_locked(self.root):raise ValueError('Stop the active run before preflight')
        from .benchmark import profile
        report=profile(self.c,self.root.parent,updates,warmup);write_json(self.root/'profile.json',report);return report
    def extend(self,stage,additional):
        if stage not in self.c.kind['stages'] or type(additional)!=int or additional<1:raise ValueError('Positive stage extension required')
        later=self.c.kind['stages'][self.c.kind['stages'].index(stage)+1:]
        if any((self.root/'stages'/s/'resume.pt').exists() for s in later):raise ValueError('A downstream stage depends on these weights; start a new experiment')
        if (self.root/'export').exists():
            import shutil,uuid
            archive=self.root/'budget-exports'/uuid.uuid4().hex;archive.mkdir(parents=True);shutil.copytree(self.root/'export',archive/'export');write_json(archive/'config.json',self.c.to_dict())
        self.c=replace(self.c,budgets={**self.c.budgets,stage:self.c.budgets[stage]+additional}).validate();return self.run(extend=True)
