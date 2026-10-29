import json
from pathlib import Path
import subprocess,sys,tarfile,venv,zipfile,re,os
import pytest


def test_notebook_structure_and_real_archive_bootstrap(tmp_path):
    notebook=json.loads(Path('notebooks/colab-legacy.ipynb').read_text())
    codes=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert len(codes)>=5
    for j,code in enumerate(codes):compile(code,f'cell-{j}','exec')
    setup=next(c for c in codes if 'SOURCE_ROOT = CACHE_ROOT' in c)
    bootstrap=setup.split('ENV_ROOT =')[0]
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('pdeflow/pyproject.toml','[project]\nname="pdeflow"\n')
    cache=tmp_path/'cache';cache.mkdir()
    scope=dict(Path=Path,json=json,subprocess=subprocess,zipfile=zipfile,tarfile=tarfile,venv=venv,re=re,os=os,sys=sys,CACHE_ROOT=cache,REPOSITORY_URL='',COMMIT_SHA='',SOURCE_ARCHIVE=str(archive))
    exec(compile(bootstrap,'notebook-bootstrap','exec'),scope)
    assert (scope['PACKAGE_ROOT']/'pyproject.toml').exists()
    exec(compile(bootstrap,'notebook-bootstrap-again','exec'),scope)
    with zipfile.ZipFile(archive,'w') as z:z.writestr('pdeflow/pyproject.toml','changed')
    with pytest.raises(ValueError,match='Cached source'):
        exec(compile(bootstrap,'notebook-bootstrap-changed','exec'),scope)
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape','bad')
    scope['SOURCE_ARCHIVE']=str(bad);scope['CACHE_ROOT']=tmp_path/'badcache';scope['CACHE_ROOT'].mkdir()
    with pytest.raises(ValueError,match='Unsafe archive'):
        exec(compile(bootstrap,'notebook-unsafe-archive','exec'),scope)


def test_online_notebook_real_local_cli_without_training_directory(tmp_path,monkeypatch):
    from dataclasses import replace
    from pdeflow.runtime import Config
    from pdeflow.cli import main
    n=json.loads(Path('notebooks/colab-legacy.ipynb').read_text())
    package=tmp_path/'source';profiles=package/'configs/experiments';profiles.mkdir(parents=True)
    c=Config.load('configs/experiments/online_heat_2d.json')
    c=replace(c,provider=replace(c.provider,calibration_count=2,validation_count=1,test_count=1),training=replace(c.training,steps=2,batch_size=2),model=replace(c.model,width=4,blocks=1))
    (profiles/'online_heat_2d.json').write_text(json.dumps(c.to_dict()))
    commands=[]
    def cli(*args):
        commands.append([str(a) for a in args]);monkeypatch.setattr(sys,'argv',['pdeflow',*[str(a) for a in args]]);main()
    output=tmp_path/'persistent';output.mkdir();cache=tmp_path/'cache';cache.mkdir()
    scope={'Path':Path,'json':json,'PACKAGE_ROOT':package,'OUTPUT_ROOT':output,'CACHE_ROOT':cache,'cli':cli}
    setup=''.join(n['cells'][3]['source']).replace('PROFILE = "cpu_2d"','PROFILE = "online_heat_2d"')
    exec(compile(setup,'profile','exec'),scope)
    launch=''.join(n['cells'][4]['source']);exec(compile(launch,'online-launch','exec'),scope)
    assert not scope['DATA'].exists() and scope['PERSISTENT_EVALUATION'].is_dir()
    assert (scope['EXPORT']/'inference.pt').exists()
    assert '--data' not in next(cmd for cmd in commands if cmd[0]=='train')
    import shutil
    shutil.rmtree(scope['EVALUATION']) # disposable runtime cache; persistent held-outs remain
    exec(compile(launch,'online-resume','exec'),scope)
    assert (scope['EVALUATION']/'manifest.json').exists()
    assert '--resume' in next(cmd for cmd in reversed(commands) if cmd[0]=='train')
