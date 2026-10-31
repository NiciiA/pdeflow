"""Fixed-effective-batch A1/A2/B accumulated vs actual8/16/32 microbatch sweep.

Fresh subprocesses isolate allocator peaks. Cold/warm trials replay the same seed
and immutable A1/A2 dependencies; only valid t=0 target files warm the replay.
No quality gate is disabled and no production Session training action is invoked.
"""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def worker(job):
    import torch,numpy as np
    from pdeflow.coordinate.contracts import CoordinateConfig
    from pdeflow.coordinate.training import train,read_checkpoint
    from pdeflow.coordinate.profiling import distribution
    from pdeflow.runtime import write_json,synchronize,RuntimeSpec
    c=CoordinateConfig(**job['config']).validate();output=Path(job['output']);stage=job['stage'];device,_=RuntimeSpec(**c.runtime).resolve()
    if job.get('cache_from'):
        original=Path(job['cache_from'])/'latent-targets'
        # Root must be empty at trainer entry. Copy reusable files when the cache
        # directory is first made, after root ownership has been validated.
        import pdeflow.coordinate.engine as engine
        original_class=engine.FrozenTargets
        def create(identity,limit,timings,root=None,prefetch=1):
            if root and original.exists():
                source=original/Path(root).name
                if source.exists():shutil.copytree(source,root,dirs_exist_ok=True)
            return original_class(identity,limit,timings,root,prefetch)
        engine.FrozenTargets=create
    start=time.perf_counter()
    try:
        r=train(c.to_dict(),output,stage,job.get('dependency'));synchronize(device);wall=time.perf_counter()-start
        cp,_=read_checkpoint(r['export']);rows=cp['history'];measured=rows[job['warmup']:]
        times=[x['end_to_end_seconds'] for x in measured];data=[x['component_timings'] for x in rows]
        result={'status':'completed','config':c.to_dict(),'source':cp['scientific_source'],'dependency':cp['dependency'],'normalizations':cp['normalizations'],'seed':c.seed,'precision':cp['actual_precision'],
          'all_in_stage_seconds':wall,'setup_seconds':r['setup_seconds'],'cache_identity_hash_seconds_setup':r.get('cache_identity_hash_seconds_setup'),
          'final_calibration_and_export_seconds':r['final_persistence_and_calibration_seconds'],'warmup_rows':rows[:job['warmup']],'measured_rows':measured,
          'end_to_end_seconds':distribution(times),'update_seconds':distribution([x['step_seconds'] for x in measured]),'first_update_seconds':rows[0]['end_to_end_seconds'],
          'examples_per_second':c.batch_size/(sum(times)/len(times)),'all_in_examples_per_second':c.batch_size*len(rows)/wall,
          'peak_memory':r['memory'],'cache_bytes':sum(p.stat().st_size for p in output.rglob('*.pt') if 'latent-targets' in p.parts),
          'frozen_preparation':rows[-1]['frozen_preparation'],'skipped_updates':sum(x['optimizer_update_skipped'] for x in rows),
          'preprocessing_exclusive_host_seconds':{key:sum(x['host_exclusive_seconds'].get(key,0.) for x in data) for key in ('frozen_key_construction','frozen_cache_lookup','frozen_reference_wait','frozen_reference_evaluation','frozen_disk_read','frozen_deserialization','frozen_output_transfer','frozen_serialization','frozen_disk_write','reference_evaluation','region_analysis','transfer')},
          'timing_scope':'All-in stage includes setup/final calibration/export. End-to-end updates include scheduled validation/checkpoints. Prefetched reference worker time overlaps owner compute; separately reported, never added to wall.'}
    except torch.OutOfMemoryError as error:
        result={'status':'out_of_memory','config':c.to_dict(),'error':str(error),'meaning':'Requested variant failed; no capacity/points/objective shrink applied.'}
    write_json(output/'benchmark.json',result);print(json.dumps({'stage':stage,'output':str(output),'status':result['status'],'end_to_end_seconds':result.get('end_to_end_seconds'),'peak_memory':result.get('peak_memory')}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path);p.add_argument('--output',type=Path);p.add_argument('--effective-batch',type=int,default=None);p.add_argument('--microbatches',type=int,nargs='+',default=[8,16,32]);p.add_argument('--updates',type=int,default=7);p.add_argument('--warmup',type=int,default=2);p.add_argument('--worker',type=Path,help=argparse.SUPPRESS);a=p.parse_args()
    if a.worker:return worker(json.loads(a.worker.read_text()))
    if not a.config or not a.output:p.error('--config and --output required')
    if a.updates<a.warmup+5 or a.warmup<1:p.error('Use >=1 warmup and >=5 measured updates')
    from dataclasses import replace
    from pdeflow.coordinate.contracts import CoordinateConfig
    from pdeflow.coordinate.optimization import precision
    from pdeflow.colab import hardware
    from pdeflow.runtime import write_json,RuntimeSpec,digest
    requested=CoordinateConfig(**json.loads(a.config.read_text())).validate();effective=a.effective_batch or requested.batch_size
    if min(a.microbatches)<1 or max(a.microbatches)>effective:p.error('Every microbatch must fit the SAME effective batch; explicitly use --effective-batch32 to compare8/16/32 if the requested batch is8')
    device,_=RuntimeSpec(**requested.runtime).resolve();policy,_=precision(requested.training_options.get('precision','fp32'),device)
    common=replace(requested,batch_size=effective,ae_steps=a.updates,evolution_steps=a.updates,flow_steps=a.updates,checkpoint_every=a.updates,
        training_options={**requested.training_options,'precision':policy,'profile_components':True,'microbatch':min(a.microbatches),'accumulation':(effective+min(a.microbatches)-1)//min(a.microbatches),'validation_every':a.updates})
    common.validate();a.output.mkdir(parents=True,exist_ok=False)
    report={'hardware':hardware(common.runtime['device'],a.output),'requested_config':requested.to_dict(),'common_probe_config':common.to_dict(),'explicit_effective_batch_override':a.effective_batch,'status':'running','results':{},'meaning':'Same architecture/points/loss/precision/splits/distributions and effective batch for EVERY comparison variant. Separate experiment if effective batch differs from original profile. Cold means empty target cache; warm means deterministic replay with only valid initial-state target files, no fixed continuous-time frame set. Shared2-update random A1/A2 anchors are pipeline fixtures, never accuracy models.'}
    write_json(a.output/'sweep.json',report)
    def launch(name,config,stage,dependency=None,cache_from=None,warmup=None):
        path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);job={'config':config.to_dict(),'stage':stage,'output':str(path.resolve()),'dependency':str(Path(dependency).resolve()) if dependency else None,'warmup':a.warmup if warmup is None else warmup,'cache_from':str(Path(cache_from).resolve()) if cache_from else None}
        job_path=a.output/(name.replace('/','-')+'-job.json');write_json(job_path,job)
        with (a.output/(name.replace('/','-')+'.log')).open('w') as log:
            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(job_path.resolve())],stdout=log,stderr=subprocess.STDOUT)
        if r.returncode:raise RuntimeError('Benchmark failed; inspect '+str(log.name)+'. No failure treated as speedup.')
        return json.loads((path/'benchmark.json').read_text())
    anchor=replace(common,ae_steps=2,evolution_steps=2,flow_steps=2,checkpoint_every=2,training_options={**common.training_options,'validation_every':2})
    for stage,dep in [('autoencoder',None),('evolution',a.output/'anchors/autoencoder/autoencoder.pt')]:
        r=launch('anchors/'+stage,anchor,stage,dep,warmup=0)
        if r['status']!='completed':raise RuntimeError('Shared anchor OOM; no comparison variants launched')
    variants=[('accumulated',min(a.microbatches)),*[('batched',m) for m in a.microbatches]]
    for execution,micro in variants:
        cc=replace(common,training_options={**common.training_options,'execution':execution,'microbatch':micro,'accumulation':(effective+micro-1)//micro})
        label=f'{execution}-{micro}';report['results'][label]={}
        for stage,dep in [('autoencoder',None),('evolution',a.output/'anchors/autoencoder/autoencoder.pt'),('flow',a.output/'anchors/evolution/evolution.pt')]:
            report['results'][label][stage]={}
            for condition in ('cold','warm'):
                name=f'{label}/{condition}/{stage}';r=launch(name,cc,stage,dep,a.output/label/'cold'/stage if condition=='warm' else None)
                report['results'][label][stage][condition]=r
                write_json(a.output/'sweep.json',report)
                if r['status']!='completed':break
    report['numerical_summary']={}
    baseline=report['results'][f'accumulated-{min(a.microbatches)}']
    for label,stages in report['results'].items():
        if label.startswith('accumulated'):continue
        report['numerical_summary'][label]={}
        for stage,conditions in stages.items():
            b=baseline[stage]['cold'];x=conditions['cold']
            if x['status']!='completed' or b['status']!='completed':continue
            pairs=list(zip(b['measured_rows'],x['measured_rows']))
            report['numerical_summary'][label][stage]={
                'max_training_loss_absolute_difference':max(abs(a['training_loss']-z['training_loss']) for a,z in pairs),
                'max_gradient_norm_absolute_difference':max((abs(a['gradient_norm']-z['gradient_norm']) for a,z in pairs if a['gradient_norm'] is not None and z['gradient_norm'] is not None),default=None),
                'family_counts_match':all(a['family_example_counts']==z['family_example_counts'] for a,z in pairs),
                'skipped_updates':x['skipped_updates'],
                'meaning':'Norm agreement is an additional diagnostic, not full gradient-vector equivalence. Strict elementwise gradient tests are separate CPU checks.'}
    report['status']='complete';write_json(a.output/'sweep.json',report);print(a.output/'sweep.json')


if __name__=='__main__':main()
