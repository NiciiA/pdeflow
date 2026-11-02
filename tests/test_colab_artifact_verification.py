"""Archive/bundle verification must fail before unsafe extraction or provenance loss."""
import hashlib,json,subprocess,sys,zipfile
from pathlib import Path
TOOL=Path(__file__).parents[1]/'scripts/verify_colab_artifacts.py'
def bundle(tmp_path):
    source=tmp_path/'inputs'/'bundle';(source/'bundles').mkdir(parents=True);data=b'actual artifact';(source/'data.txt').write_bytes(data)
    (source/'bundles/diagnostic.json').write_text(json.dumps({'kind':'diagnostic','files':{'data.txt':{'sha256':hashlib.sha256(data).hexdigest()}}}))
    return source

def run(tmp_path):return subprocess.run([sys.executable,str(TOOL),'--artifacts',str(tmp_path/'inputs'),'--output',str(tmp_path/'audit')],capture_output=True,text=True)

def test_verifies_bundle_and_refuses_changed_original_baseline(tmp_path):
    source=bundle(tmp_path);assert run(tmp_path).returncode==0
    baseline=(tmp_path/'audit/artifact-inventory.json').read_bytes();(source/'data.txt').write_text('changed')
    result=run(tmp_path);assert result.returncode!=0 and 'baseline is not overwritten' in result.stderr
    assert (tmp_path/'audit/artifact-inventory.json').read_bytes()==baseline

def test_incorrect_manifest_checksum_fails(tmp_path):
    source=bundle(tmp_path);(source/'data.txt').write_text('incorrect')
    result=run(tmp_path);assert result.returncode!=0 and 'checksum-mismatched' in result.stderr
    assert (source/'data.txt').read_text()=='incorrect'

def test_archive_traversal_rejected_before_extraction(tmp_path):
    source=bundle(tmp_path)
    with zipfile.ZipFile(source/'source.zip','w') as archive:archive.writestr('../escape.txt','must not be extracted')
    assert run(tmp_path).returncode!=0
    assert not list((tmp_path/'audit').rglob('escape.txt'))
