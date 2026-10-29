"""Compile all cells and execute their actual CPU Run-all path in an isolated env."""
from pathlib import Path
import json,os,subprocess,sys,zipfile,shutil
import pytest


def test_notebook_structure():
    n=json.loads(Path('notebooks/colab.ipynb').read_text())
    assert n['nbformat']==4 and n['metadata']['accelerator']=='GPU'
    code=[c for c in n['cells'] if c['cell_type']=='code'];assert len(code)==6
    for i,c in enumerate(code):
        compile(''.join(c['source']),f'cell-{i}','exec');assert c['outputs']==[] and c['execution_count'] is None
    assert 'RUN_TRAINING = False' in ''.join(code[4]['source'])
    assert Path('notebooks/colab-legacy.ipynb').exists()


def test_real_run_all_source_archive_setup_profile_train_export(tmp_path):
    interpreter=os.environ.get('PDEFLOW_NOTEBOOK_PYTHON')
    if not interpreter:pytest.skip('Set PDEFLOW_NOTEBOOK_PYTHON to an isolated Python with real IPython and core dependencies')
    root=Path.cwd();archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('pyproject.toml','README.md','MANIFEST.in'):
            z.write(root/name,'pdeflow/'+name)
        for directory in ('src','configs','scripts'):
            for p in (root/directory).rglob('*'):
                if p.is_file() and not any(x in ('__pycache__',) or x.endswith('.egg-info') for x in p.parts) and p.suffix in ('.py','.json'):
                    z.write(p,'pdeflow/'+p.relative_to(root).as_posix())
    n=json.loads((root/'notebooks/colab.ipynb').read_text());cells=[''.join(c['source']) for c in n['cells'] if c['cell_type']=='code']
    replacements={
      'SOURCE_ARCHIVE = ""':f'SOURCE_ARCHIVE = {str(archive)!r}',
      'Path("/content/pdeflow-source")':f'Path({str(tmp_path/"source-cache")!r})',
      'USE_DRIVE = True':'USE_DRIVE = False',
      'PERSISTENT_ROOT = "/content/drive/MyDrive/PDEFlow-training"':f'PERSISTENT_ROOT = {str(tmp_path/"persistent")!r}',
      'LOCAL_ROOT = "/content/pdeflow-training"':f'LOCAL_ROOT = {str(tmp_path/"fast")!r}',
      'RUN_NAME = "heat-large-01"':'RUN_NAME = "cpu-notebook"',
      'PRESET = "large"':'PRESET = "pipeline-check"',
      'DEVICE = "cuda"':'DEVICE = "cpu"',
      'PRECISION = "auto"':'PRECISION = "fp32"',
      'OVERRIDE_FAILED_GATE = False':'OVERRIDE_FAILED_GATE = True',
      'ALLOW_CPU_PIPELINE_CHECK = False':'ALLOW_CPU_PIPELINE_CHECK = True',
      'RUN_TRAINING = False':'RUN_TRAINING = True',
      'DOWNLOAD_MODEL = True':'DOWNLOAD_MODEL = False'}
    script='';executed=[]
    for i,cell in enumerate(cells):
        for before,after in replacements.items():cell=cell.replace(before,after)
        executed.append(cell)
        script+=f'print("EXECUTING NOTEBOOK CELL {i+1}",flush=True)\n'+cell+'\n'
    script+='assert (session.root / "export/coordinate.pt").exists()\nassert outcome["state"] == "completed"\n'
    # Real repeated setup after model imports must reuse the same source safely.
    script+=f'exec({executed[0]!r})\n'
    script+='cached_readme=PACKAGE_ROOT/"README.md"\noriginal_readme=cached_readme.read_bytes()\ncached_readme.write_bytes(b"changed cached source")\n'
    script+=f'try:\n    exec({executed[0]!r})\nexcept ValueError as error:\n    assert "Cached source changed" in str(error)\nelse:\n    raise AssertionError("Modified source cache was silently reused")\nfinally:\n    cached_readme.write_bytes(original_readme)\n'
    runner=tmp_path/'run_all.py';runner.write_text(script)
    result=subprocess.run([interpreter,str(runner)],cwd=tmp_path,env={**os.environ,'MPLBACKEND':'Agg'},capture_output=True,text=True,timeout=180)
    evidence=Path(os.environ.get('PDEFLOW_NOTEBOOK_EVIDENCE',str(tmp_path/'evidence')));evidence.mkdir(parents=True,exist_ok=True)
    (evidence/'notebook-run-all.log').write_text(result.stdout+'\nSTDERR\n'+result.stderr)
    assert result.returncode==0,result.stdout[-3000:]+result.stderr[-5000:]
    shutil.copytree(tmp_path/'fast/cpu-notebook',evidence/'notebook-run',dirs_exist_ok=True)
    assert 'EXECUTING NOTEBOOK CELL 6' in result.stdout and 'Workflow: completed' in result.stdout
