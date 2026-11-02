"""Compile eight steps; execute real bounded CPU cells in a separate interpreter."""
from pathlib import Path
import json,os,subprocess,sys,shutil
import pytest
from pdeflow.source_bundle import package


def test_notebook_structure():
    n=json.loads(Path('notebooks/legacy-factorized-colab.ipynb').read_text());assert n['nbformat']==4 and n['metadata']['accelerator']=='GPU'
    code=[c for c in n['cells'] if c['cell_type']=='code'];assert len(code)==8
    for i,c in enumerate(code):
        compile(''.join(c['source']),f'cell-{i}','exec');assert c['outputs']==[] and c['execution_count'] is None
    assert 'RUN_TRAINING = False' in ''.join(code[5]['source'])
    assert 'Session.from_store' in ''.join(code[2]['source'])
    assert 'session.finalize' in ''.join(code[6]['source'])
    assert 'override_gate=False' in ''.join(code[5]['source'])


@pytest.mark.parametrize('field_loss',['mse','region_balanced_mse'])
def test_real_run_all_source_storage_train_save_restore(tmp_path,field_loss):
    interpreter=os.environ.get('PDEFLOW_NOTEBOOK_PYTHON')
    if not interpreter:pytest.skip('Set PDEFLOW_NOTEBOOK_PYTHON to isolated Python with real IPython and core dependencies')
    root=Path.cwd();archive=tmp_path/'source.zip';package(archive)
    n=json.loads((root/'notebooks/legacy-factorized-colab.ipynb').read_text());cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    replacements={
      'SOURCE_ARCHIVE = ""':f'SOURCE_ARCHIVE = {str(archive)!r}',
      'Path("/content/pdeflow-source")':f'Path({str(tmp_path/"source-cache")!r})',
      'ENABLE_DRIVE_PERSISTENCE = True':'ENABLE_DRIVE_PERSISTENCE = False',
      'DRIVE_RUNS_ROOT = "/content/drive/MyDrive/PDEFlow-training"':f'DRIVE_RUNS_ROOT = {str(tmp_path/"persistent")!r}',
      'LOCAL_ROOT = "/content/pdeflow-training"':f'LOCAL_ROOT = {str(tmp_path/"fast")!r}',
      'PRESET = "medium"':'PRESET = "pipeline-check"',
      'DEVICE = "cuda"':'DEVICE = "cpu"',
      'PRECISION = "auto"':'PRECISION = "fp32"',
      'ALLOW_CPU_PIPELINE_CHECK = False':'ALLOW_CPU_PIPELINE_CHECK = True',
      'RUN_TRAINING = False':'RUN_TRAINING = True',
      'STAGES = "both"':'STAGES = "representation"',
      'CONDITIONAL_EXPERIMENT = "B0"':'CONDITIONAL_EXPERIMENT = "B3"',
      'EFFECTIVE_BATCH = 8':'EFFECTIVE_BATCH = 2',
      'APPLY_BATCH_SETTINGS = False':'APPLY_BATCH_SETTINGS = True',
      'INSTALL_PROJECT = True':'INSTALL_PROJECT = False',
      'FIELD_LOSS = "mse"':f'FIELD_LOSS = {field_loss!r}'}
    executed=[];script=''
    for i,cell in enumerate(cells):
        for before,after in replacements.items():cell=cell.replace(before,after)
        executed.append(cell);script+=f'print("EXECUTING NOTEBOOK STEP {i+1}",flush=True)\n'+cell+'\n'
    script+=f'assert session.c.field_loss["type"] == {field_loss!r}\n'
    script+='from pdeflow.coordinate.training import read_checkpoint\nassert read_checkpoint(session.root/"autoencoder/resume.pt")[0]["resolved_field_loss"]["settings"]["type"] == session.c.field_loss["type"]\n'
    script+='assert session.c.batch_size == 2\nassert (session.root / "autoencoder/resume.pt").exists()\nassert not (session.root / "flow/resume.pt").exists()\nassert saved["artifact_views"]["inference"]["available"] is False\n'
    # Mounting Drive itself cannot be tested here. Exercise the same public local
    # persistent store and restore into an empty root, with no mocked numerics.
    script+='session.persistence_enabled=True\nsession.sync(True)\n'
    script+=f'restored=Session.from_store(DRIVE_RUNS_ROOT,session.name,{str(tmp_path/"fresh")!r},device="cpu",allow_cpu_check=True)\n'
    script+='assert restored.c.to_dict()==session.c.to_dict()\nassert (restored.root/"source/source.zip").exists()\nprint("FRESH RUNTIME RESTORE OK")\n'
    script+=f'exec({executed[0]!r})\n'
    script+='cached_readme=PACKAGE_ROOT/"README.md"\noriginal_readme=cached_readme.read_bytes()\ncached_readme.write_bytes(b"changed cached source")\n'
    script+=f'try:\n    exec({executed[0]!r})\nexcept ValueError as error:\n    assert "Cached source changed" in str(error)\nelse:\n    raise AssertionError("Modified cache silently reused")\nfinally:\n    cached_readme.write_bytes(original_readme)\n'
    runner=tmp_path/'run_all.py';runner.write_text(script)
    result=subprocess.run([interpreter,str(runner)],cwd=tmp_path,env={**os.environ,'MPLBACKEND':'Agg'},capture_output=True,text=True,timeout=180)
    evidence=Path(os.environ.get('PDEFLOW_NOTEBOOK_EVIDENCE',str(tmp_path/'evidence')))/field_loss;evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'notebook-run-all.log').write_text(result.stdout+'\nSTDERR\n'+result.stderr)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr[-5000:]
    assert 'EXECUTING NOTEBOOK STEP 8' in result.stdout and 'FRESH RUNTIME RESTORE OK' in result.stdout
    shutil.copytree(tmp_path/'fast',evidence/'notebook-run',dirs_exist_ok=True)
