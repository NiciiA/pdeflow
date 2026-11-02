"""Explicit NEW-experiment shared physical-module initialization, never resume migration."""
from pathlib import Path
import torch
from pdeflow.runtime import file_digest,digest,write_json,read_json,atomic_write
from .provenance import read_checkpoint,scientific_source


def initialize(c,root,model,metadata):
    root=Path(root);payload=root/'bootstrap/initialization.pt';manifest=root/'bootstrap/initialization.json'
    if payload.exists():
        record=read_json(manifest)
        if file_digest(payload)!=record['sha256'] or record['config_identity']!=digest(c.to_dict()):raise ValueError('Warm initialization changed')
        saved=torch.load(payload,map_location='cpu',weights_only=True);model.load_state_dict(saved['model'],strict=True);return record
    parent=Path(c.warm_start['checkpoint']);cp,prior=read_checkpoint(parent)
    if not cp['inference_ready'] or cp['architecture']!='mesh_m2_v1':raise ValueError('Shared physical warm start requires a trained M2 standalone export')
    if cp['scientific_source']['sha256']!=scientific_source()['sha256']:raise ValueError('Shared mesh warm start source differs; use recorded compatible source')
    for name in ('width','depth','heads','node_shape','coarse_shape','global_tokens','neighbors','domain','interval','family_weights'):
        if getattr(c,name)!=getattr(prior,name):raise ValueError('Warm-start physical contract differs: '+name)
    if cp['dataset_identity']!=metadata['sha256'] or digest(cp['normalizations'])!=digest(metadata['normalizations']):raise ValueError('Warm-start dataset/normalization mismatch')
    if not c.pipeline_check and not (cp.get('validation') or {}).get('passed'):raise ValueError('Warm parent failed its minimum validation gate; do not bypass it')
    state=model.state_dict();copied=[];new=[]
    for name,value in state.items():
        allowed=name.startswith(('conditioner.','decoder.','evolution.')) or c.architecture=='mesh_m3_fgn_v1' and name.startswith('state_head.')
        if allowed and name in cp['model']:
            if value.shape!=cp['model'][name].shape:raise ValueError('Warm tensor shape mismatch: '+name)
            state[name]=cp['model'][name];copied.append(name)
        else:new.append(name)
    model.load_state_dict(state,strict=True);payload.parent.mkdir(parents=True);atomic_write(payload,lambda f:torch.save({'kind':'mesh_explicit_warm_initialization','model':model.state_dict(),'inference_ready':False},f));record={'kind':'new_experiment_not_resume','sha256':file_digest(payload),'config_identity':digest(c.to_dict()),'parent_sha256':file_digest(parent),'parent_architecture':cp['architecture'],'copied_keys':copied,'newly_initialized_keys':new,'parent_training_provenance':cp.get('training_provenance',{}),'optimizer':'Fresh; all declared new-run stages/budgets are retained and counted','cost_warning':'Warm comparisons include parent training cost; not a from-scratch generative-method ablation'};write_json(manifest,record);return record
