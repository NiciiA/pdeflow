"""Small versioned B ablations; no changes to data, time policies or A1/A2."""
from dataclasses import replace

EXPERIMENTS={
    'B0': 'Baseline',
    'B1': 'Spatial conditioning',
    'B2': 'Observation-aligned',
    'B3': 'Spatial + observation-aligned',
}


def conditional_experiment(c,name='B0',auxiliary=None,conditioning=None):
    if name not in EXPERIMENTS:raise ValueError('Conditional experiment must be B0, B1, B2 or B3')
    spatial=name in ('B1','B3');aligned=name in ('B2','B3')
    # .01 is an experimental starting weight, not a validated optimum.
    aux={'beta':.01,'minimum_s':.5,'maximum_s':1.,'sensor_budget':8,'query_chunk':8,'discrepancy_relative':.05,'warmup_updates':100,**(auxiliary or {})} if aligned else {}
    reader={'version':'spatial_v1' if spatial else 'pooled_v1',**(conditioning or {})}
    if reader['version']!=('spatial_v1' if spatial else 'pooled_v1'):raise ValueError('Conditioning override contradicts named experiment')
    return replace(c,joint={**c.joint,'conditioning':reader,'observation_auxiliary':aux}).validate()


def copy_frozen_dependencies(parent,destination):
    """Copy exact canonical A1/A2 + data recipe and their original source.

    New B experiments own complete, relocatable diagnostic/resumable artifacts.
    No dependency hashes or checkpoint paths are rewritten; layouts are retained.
    """
    from pathlib import Path
    import shutil
    from pdeflow.runtime import file_digest,write_json
    from .training import read_checkpoint
    from pdeflow.source_bundle import verify
    parent=Path(parent).resolve();destination=Path(destination).resolve()
    for stage in ('autoencoder','evolution'):
        if (destination/stage).exists():raise FileExistsError('Frozen stage already exists: '+stage)
    original=parent/'source/source.zip';verify(original);inventory={}
    for stage in ('autoencoder','evolution'):
        export=stage+'.pt';cp,_=read_checkpoint(parent/stage/export,stage)
        if not cp['completed']:raise ValueError('Frozen stage must be completed')
        (destination/stage).mkdir(parents=True)
        for path in (parent/stage).iterdir():
            if path.is_file() and (path.name in (export,'resume.pt','history-live.jsonl') or path.suffix=='.json' or path.name.endswith('-preview.npz')):
                before=file_digest(path);target=destination/stage/path.name;shutil.copy2(path,target)
                if file_digest(target)!=before or file_digest(path)!=before:raise ValueError('Dependency copy checksum mismatch')
                inventory[str(path.relative_to(parent))]=before
    (destination/'frozen-source').mkdir(exist_ok=False);shutil.copy2(original,destination/'frozen-source/source.zip')
    inventory['source/source.zip']=file_digest(original)
    write_json(destination/'frozen-dependencies.json',{'parent':str(parent),'files':inventory,'policy':'Exact copies, including matching encoder, optimizer states and original source; new B optimizer is independent'})
    return inventory
