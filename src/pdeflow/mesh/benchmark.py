"""Disposable actual-candidate/objective preflight; never changes a requested workload."""
from dataclasses import replace
from pathlib import Path
import tempfile,time,statistics
from pdeflow.runtime import digest,read_json
from .contracts import MeshConfig
from .training import train_stage
from .provenance import scientific_source


def profile(c,local_root,updates=8,warmup=2):
    if updates<warmup+5 or warmup<1:raise ValueError('At least1 warmup and5 measured updates required')
    from pdeflow.colab import hardware
    from pdeflow.coordinate.random_state import capture,restore
    import torch
    before=capture();cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    report={'requested_config':digest(c.to_dict()),'scientific_source':scientific_source()['sha256'],'hardware':hardware(c.runtime['device']),'warmup_updates':warmup,'measured_updates':updates-warmup,'stages':{},'success':False,'scope':'Same candidate, architecture, points, objective, batch and precision. Disposable budgets/checkpoint/validation cadence only. No accuracy result.'}
    try:
        with tempfile.TemporaryDirectory(prefix='mesh-preflight-',dir=local_root) as directory:
            pc=replace(c,budgets={s:updates for s in c.budgets},checkpoint_every=updates,validation_every=updates);dependency=None
            for stage in c.kind['stages']:
                record=train_stage(pc,directory,stage,dependency);rows=read_json(Path(directory)/'stages'/stage/'history.json');measured=[r for r in rows if not r['optimizer_update_skipped']][warmup:]
                seconds=[r['step_seconds'] for r in measured];reference=[r['reference_host_seconds'] for r in measured]
                report['stages'][stage]={'warmup_rows':rows[:warmup],'median_seconds_per_successful_update':statistics.median(seconds),'minimum':min(seconds),'maximum':max(seconds),'p90':sorted(seconds)[min(len(seconds)-1,int(.9*len(seconds)))],'examples_per_second':c.batch_size/statistics.median(seconds),'median_reference_host_seconds':statistics.median(reference),'cold_first_update_seconds':rows[0]['step_seconds'],'setup_seconds':record['setup_seconds'],'training_wall_seconds':record['training_seconds'],'memory':record['memory'],'actual_forward_microbatch':measured[0]['actual_forward_microbatch'],'skipped_updates':record['skipped_updates'],'rough_training_seconds_excluding_setup_evaluation_export':statistics.median(seconds)*c.budgets[stage],'measured_rows':measured}
                dependency=Path(directory)/'stages'/stage/'resume.pt'
            report['success']=True
    finally:
        restore(before);torch.set_rng_state(cpu)
        if cuda:torch.cuda.set_rng_state_all(cuda)
    return report
