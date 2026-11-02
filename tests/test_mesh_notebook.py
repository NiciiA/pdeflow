"""Run the thin new notebook from a freshly verified source extraction on CPU."""
from pathlib import Path
import os,json,subprocess,sys
import pytest
from pdeflow.source_bundle import package


def test_mesh_notebook_structure():
    n=json.loads(Path('notebooks/colab.ipynb').read_text());cells=[c for c in n['cells'] if c['cell_type']=='code'];assert len(cells)==8
    for i,c in enumerate(cells):compile(''.join(c['source']),f'cell-{i}','exec');assert c['outputs']==[] and c['execution_count'] is None
    assert 'RUN_TRAINING = False' in ''.join(cells[5]['source']) and 'override_gate' not in ''.join(cells[5]['source'])
    assert Path('notebooks/legacy-factorized-colab.ipynb').is_file()

@pytest.mark.parametrize('suite',[False,True])
def test_mesh_notebook_cpu_fresh_source_all_cells(tmp_path,suite):
    root=Path.cwd();package(tmp_path/'source.zip');notebook=json.loads(Path('notebooks/colab.ipynb').read_text());code=[ ''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code'];script=''
    changes={'SOURCE_ARCHIVE = ""':f'SOURCE_ARCHIVE = {str(tmp_path/"source.zip")!r}','Path("/content/pdeflow-source")':f'Path({str(tmp_path/"source-cache")!r})','ENABLE_DRIVE_PERSISTENCE = True':'ENABLE_DRIVE_PERSISTENCE = False','DRIVE_RUNS_ROOT = "/content/drive/MyDrive/PDEFlow-mesh"':f'DRIVE_RUNS_ROOT = {str(tmp_path/"store")!r}','LOCAL_ROOT = "/content/pdeflow-mesh"':f'LOCAL_ROOT = {str(tmp_path/"local")!r}','CAPACITY = "medium"':'CAPACITY = "pipeline-check"','DEVICE = "cuda"':'DEVICE = "cpu"','INSTALL_PROJECT = True':'INSTALL_PROJECT = False','RUN_TRAINING = False':'RUN_TRAINING = True','EFFECTIVE_BATCH = 8':'EFFECTIVE_BATCH = 2','FORWARD_MICROBATCH = 8':'FORWARD_MICROBATCH = 2','SUITE = False':f'SUITE = {suite}','ADVANCED = {}':'ADVANCED = {"data":{"query_points":8}}'}
    for i,cell in enumerate(code):
        for a,b in changes.items():cell=cell.replace(a,b)
        script+=f'print("CELL {i+1}",flush=True)\n'+cell+'\n'
    script+='session.persistent=Path(DRIVE_RUNS_ROOT)/session.root.name\nsession.sync("notebook-test")\n'
    script+=f'restored=Suite.from_store(DRIVE_RUNS_ROOT,session.root.name,{str(tmp_path/"restored")!r}) if SUITE else MeshSession.from_store(DRIVE_RUNS_ROOT,session.root.name,{str(tmp_path/"restored")!r},device="cpu")\n'
    script+='assert restored.run()["state"]=="completed"\nprint("FRESH SOURCE / VERIFIED RESTORE OK")\n'
    runner=tmp_path/'run_all.py';runner.write_text(script);result=subprocess.run([sys.executable,str(runner)],cwd=tmp_path,env={**os.environ,'MPLBACKEND':'Agg'},text=True,capture_output=True,timeout=180)
    evidence=root/'outputs/mesh-family-20261102'/('notebook-suite' if suite else 'notebook-single');evidence.mkdir(exist_ok=True);(evidence/'execution.txt').write_text(result.stdout+'\nSTDERR\n'+result.stderr)
    assert result.returncode==0,result.stdout[-2000:]+result.stderr[-4000:]
    assert 'CELL 8' in result.stdout and 'FRESH SOURCE / VERIFIED RESTORE OK' in result.stdout
