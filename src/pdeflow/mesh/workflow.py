"""Small adapter between managed UI workflows and the shared mesh lifecycle."""
from dataclasses import replace
from pathlib import Path
from pdeflow.runtime import RuntimeSpec,read_json,write_json
from .contracts import preset,MeshConfig,CANDIDATES
from .session import MeshSession
from .provenance import counts


def plan(request):
    from pdeflow.training_workflow import BUDGETS
    architecture=request.get('model','mesh_m1_v1');budget=request.get('budget','pipeline-check');problem=request.get('problem','heat-wave-2d');advanced=dict(request.get('advanced') or {})
    if architecture not in CANDIDATES or budget not in BUDGETS or problem not in ('heat-2d','wave-2d','heat-wave-2d'):raise ValueError('Choose a supported mesh candidate, heat/wave family and budget')
    capacity=advanced.pop('capacity','pipeline-check' if budget=='pipeline-check' else 'xsmall');device,_=RuntimeSpec(device=request.get('device','auto')).resolve();c=preset(capacity,architecture,str(device),'auto');weights={'heat':.5,'wave':.5} if problem=='heat-wave-2d' else {problem.split('-')[0]:1.};updates=2 if budget=='pipeline-check' else 500 if budget=='development' else 10000
    raw=c.to_dict();raw.update(family_weights=weights,budgets={s:updates for s in c.budgets},pipeline_check=budget=='pipeline-check',checkpoint_every=min(50,updates),validation_every=min(50,updates))
    if budget=='development':raw['data']={**raw['data'],'train_count':256,'validation_count':24,'test_count':24}
    settings={k:advanced.pop(k) for k in ('learning_rate','batch_size','microbatch','stage_settings','batch_policy') if k in advanced}
    if any(settings.get(k,'absent') is None for k in ('learning_rate','batch_size','microbatch')):raise ValueError('Learning rate, effective batch and microbatch cannot be empty; enter explicit valid settings')
    for key,value in advanced.items():
        if key not in raw or key in ('architecture','runtime','pipeline_check','batch_reference'):raise ValueError('Unsupported mesh advanced setting: '+key)
        raw[key]={**raw[key],**value} if isinstance(value,dict) and isinstance(raw[key],dict) else value
    c=MeshConfig(**raw).validate()
    from .settings import configure,summary
    c=configure(c,learning_rate=settings.get('learning_rate'),effective_batch=settings.get('batch_size'),microbatch=settings.get('microbatch'),stages=settings.get('stage_settings'),budget_policy=settings.get('batch_policy','keep_updates'))
    if set(c.family_weights)!=set(weights):raise ValueError('Training equations conflict with family_weights; no family fallback')
    from pdeflow.coordinate.physics import training_spec
    from pdeflow.coordinate.training import data_config
    from pdeflow.coordinate.dataset import recipe
    physics=training_spec(c.reference_config());r=c.reference_config();parameters=counts(c)
    return {'schema_version':2,'training_settings':summary(c),'request':{**request,'model':architecture},'model':architecture,'model_label':c.kind['name'],'problem':problem,'budget':budget,'budget_meaning':BUDGETS[budget],'pipeline_check':c.pipeline_check,'config':c.to_dict(),'resolved_device':str(device),'grid':data_config(r).to_dict()['grid'],'training_physics':physics,'scope':'Initial-state u0/v0 only; later measurement timestamps are supplied conditioning' if not c.kind['trajectory'] else 'Complete inferred initial state and learned continuous physical evolution','stages':['prepare',*c.kind['stages'],'model_validation','register'],'updates':{s:c.budgets[s] for s in c.kind['stages']},'examples':{'training':r.train_count,'validation':r.validation_count,'untouched_test':r.test_count},'distribution':recipe(r),'equation':' · '.join(physics['equations'].values()),'architecture_parameters':parameters,'quality_policy':{'max_mse_ratio':c.gate_ratio,'pipeline_exemption':c.pipeline_check,'override':False},'field_reconstruction_loss':{'settings':c.field_loss,'stages':list(c.kind['stages']),'meaning':'Deterministic decoded field/central branch and physical dynamics, not latent velocity weighting'}}


def execute(path,record,event=None):
    config=MeshConfig(**record['plan']['config']).validate();session=MeshSession(config,path,save_updates=config.checkpoint_every)
    cached=Path(record['workspace'])/'mesh-profiles'/(__import__('pdeflow.runtime',fromlist=['digest']).digest(config.to_dict())+'.json')
    if cached.exists() and not (session.root/'profile.json').exists():__import__('shutil').copy2(cached,session.root/'profile.json')
    outcome=session.run(owned=True,event=event,extend=record.get('mesh_extend',False));artifact=session.finalize();return outcome,artifact


def list_models(workspace=None,problem=None):
    from .registry import runs
    from .inference import compatible
    from pdeflow.coordinate.physics import problem as physical_problem
    from pdeflow.coordinate.training import data_config
    result=[]
    for entry in runs(workspace):
        if not entry['available']:continue
        c=MeshConfig(**entry['config']).validate();family=problem.family if problem and problem.family in c.family_weights else next(iter(c.family_weights));r=c.reference_config();bounds=r.diffusivity_range if family=='heat' else r.wave_speed_range;physical=physical_problem(data_config(r).grid,family,sum(bounds)/2);reason='No current problem';valid=False
        if problem:
            try:compatible(c,problem);valid=True;reason='Physical support matches; M1 is initial-only' if not c.kind['trajectory'] else 'Physical support matches; accuracy unestablished'
            except ValueError as error:reason=str(error)
        result.append({'id':entry['id'],'model':c.architecture,'architecture':c.architecture,'type':c.kind['name'],'problem':physical.to_dict(),'updates':{k:v['successful_updates'] for k,v in entry.get('training_provenance',{}).items()} or {'training':entry['progress']},'status':entry['state'],'path':entry['export'],'compatibility':{'compatible':valid,'summary':reason},'available':True,'initial_only':not c.kind['trajectory'],'generative':c.kind['generative'],'pipeline_check':c.pipeline_check,'validation':None,'quality':'Pipeline-check; scientifically unevaluated' if c.pipeline_check else 'Scientifically unevaluated; inspect held-out metrics','budget':'recorded mesh configuration'})
    return result
