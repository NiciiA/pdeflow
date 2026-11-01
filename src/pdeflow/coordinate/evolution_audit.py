"""Read-only checkpoint dynamics comparisons with the same encoded-initial baseline."""
from pathlib import Path
import torch
from pdeflow.runtime import write_json,file_digest
from .training import read_checkpoint
from .model import Autoencoder
from .dataset import example
from .optimization import options,scientific_source
from .evolution_monitor import evaluate,indices


def diagnose(checkpoint,output,device='cpu',earlier=(),per_family=1,resolution=16):
    from pdeflow.runtime import RuntimeSpec
    if not 1<=per_family<=3:raise ValueError('Use1..3 fixed validation examples per family')
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new diagnostic output; existing evidence is never replaced')
    cp,c=read_checkpoint(checkpoint,'evolution');dev,_=RuntimeSpec(device=device,threads=2).resolve()
    current_source=scientific_source();critical=('coordinate/model.py','coordinate/graph.py','coordinate/dataset.py','numerics/dynamic.py','heat_continuous.py')
    changed=[name for name in critical if cp.get('scientific_source',{}).get('files',{}).get(name)!=current_source['files'][name]]
    if changed:raise ValueError('Recorded model/reference mathematics differs for '+', '.join(changed)+'. Use the exact recorded source to evaluate; no mismatched reference or operator was substituted.')
    items=[example(c,'validation',i) for i in indices(c,per_family)]
    result={'stage':'A2','checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':file_digest(checkpoint),'successful_updates':cp['progress'],'attempted_updates':cp.get('attempted_updates'),'skipped_updates':cp.get('skipped_updates'),'planned_updates':c.evolution_steps,'resolved_config':cp['config'],'field_loss':c.field_loss,'optimizer':cp.get('scheduler',{}).get('settings',options(c)['evolution']),'actual_training_precision':cp.get('actual_precision'),'evaluation_device':str(dev),'source_recorded':cp.get('scientific_source'),'source_executed':scientific_source(),'history':cp.get('history',[]),'comparisons':[],'scope':'Procedural validation examples regenerated from recorded recipe/split/seed; checkpoint training normalizations retained. No training, optimizer mutation, gate override or test tuning.'}
    for path in [*earlier,checkpoint]:
        state,other=read_checkpoint(path,'evolution')
        if state['dataset_identity']!=cp['dataset_identity'] or state['normalizations']!=cp['normalizations'] or state['dependency']['sha256']!=cp['dependency']['sha256']:raise ValueError('A2 comparison must use identical data, normalization and frozen A1 dependency')
        changed=[name for name in critical if state.get('scientific_source',{}).get('files',{}).get(name)!=current_source['files'][name]]
        if changed:raise ValueError('Earlier checkpoint mathematics differs: '+', '.join(changed))
        model=Autoencoder(other).to(dev).eval().requires_grad_(False);model.load_state_dict(state['autoencoder'])
        report,_=evaluate(c,model,cp['normalizations'],items,resolution=resolution,include_oracle=True,include_physics=True)
        result['comparisons'].append({'checkpoint':str(path),'sha256':file_digest(path),'weight_update':state['progress'],'resolved_evolution':other.evolution,'field_loss':other.field_loss,'optimizer':state.get('scheduler',{}).get('settings'),**report})
    write_json(output,result);return result
