"""Focused numerical, isolation and permanent deletion safety regressions."""
from pathlib import Path
from dataclasses import replace
import os
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from pdeflow.workbench.playground import Heat,Measurements,reference,measurements
from pdeflow.reference import load
from pdeflow.runtime import write_json,read_json,file_digest
from pdeflow.local import run_lock
from pdeflow import model_storage as storage


@pytest.mark.parametrize('kx,ky,phase',[(0,0,.7),(0,2,.3),(3,1,-.5)])
def test_cosine_fft_independent_nonunit_domain_and_constant(tmp_path,kx,ky,phase):
    cfg=Heat(lengths=[2.,3.],origins=[-1.,.4],kx=kx,ky=ky,phix=phase,phiy=.4)
    r=reference(cfg,tmp_path);p,a,_=load(tmp_path/r['id'])
    assert r['exact_max_error']<3e-14
    np.testing.assert_allclose(np.asarray(r['frames']),r['exact'],atol=3e-14)
    np.testing.assert_allclose(a[0,0].mean((1,2)),a[0,0,0].mean(),atol=1e-15)
    if kx==ky==0:np.testing.assert_allclose(a[0,0,0],a[0,0,-1])


def test_scattered_samples_exact_count_separate_noise_seed_times(tmp_path):
    cfg=Heat();r=reference(cfg,tmp_path)
    spec=Measurements(reference_id=r['id'],count=25,time=.075,noise=.03,pattern='current')
    a=measurements(spec,tmp_path);b=measurements(spec,tmp_path)
    assert len(a['rows'])==25 and a['rows']==b['rows']
    assert all(row['time']==.075 and row['sigma']==.03 for row in a['rows'])
    q=np.array([[row['x'],row['y']] for row in a['rows']])
    truth=.2+.8*np.exp(-.07*8*np.pi**2*.075)*np.cos(2*np.pi*q[:,0])*np.cos(2*np.pi*q[:,1])
    clean=np.load(tmp_path/r['id']/'samplings'/a['id']/'clean-evaluation-only.npz')['clean']
    np.testing.assert_allclose(truth,clean,atol=8e-16)
    assert not np.array_equal(clean,[row['value'] for row in a['rows']])
    for pattern in ('across','fixed'):
        c=measurements(spec.model_copy(update={'pattern':pattern}),tmp_path)
        assert len(c['rows'])==25
    assert 'clean' not in str(a.keys())


def test_random_reproducible_and_unsafe_formula_rejected(tmp_path):
    a=reference(Heat(preset='random',seed=2),tmp_path)
    b=reference(Heat(preset='random',seed=2),tmp_path)
    assert a['initial']==b['initial']
    for expr in ("__import__('os').system('ls')",'1/(x-x)','unknown+1','exp(999)'):
        with pytest.raises(ValueError):reference(Heat(preset='formula',expression=expr),tmp_path)


def weights(root,name,stage='flow',dependency=None):
    d=root/name;d.mkdir(parents=True)
    cp={'stage':stage,'autoencoder':{},'config':{},'normalization':{},'dependency':dependency or {}}
    torch.save(cp,d/'coordinate.pt');torch.save(cp,d/'resume.pt')
    write_json(d/'run.json',{'kind':'coordinate_stage_run','stage':stage,'specialist':'heat','dependency':dependency or {}})
    return d


def test_deletion_preview_cancel_hardlinks_protected_and_selection(tmp_path):
    d=weights(tmp_path,'flow');os.link(d/'coordinate.pt',d/'linked.pt')
    (d/'personal.txt').write_text('keep');(d/'unknown.pt').write_bytes(b'not a checkpoint')
    (tmp_path/'source.py').write_text('keep')
    write_json(tmp_path/'selected-model.json',{'id':file_digest(d/'coordinate.pt')[:32]})
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report=storage.inventory(tmp_path);g=report['models'][0]
    assert len(g['files'])==3 and g['selected']
    plan=storage.preview(tmp_path,all_models=True)
    assert plan['unique_bytes']==(d/'coordinate.pt').stat().st_size+(d/'resume.pt').stat().st_size
    assert all(Path(p).read_bytes()==b for p,b in before.items()) # cancel is a no-op
    with pytest.raises(ValueError):storage.apply(tmp_path,plan['id'],'yes',plan['digest'])
    result=storage.apply(tmp_path,plan['id'],plan['confirmation'],plan['digest'])
    assert not result['inventory']['models'] and not (tmp_path/'selected-model.json').exists()
    assert (d/'personal.txt').exists() and (d/'unknown.pt').exists() and (tmp_path/'source.py').exists()
    assert result['actual_unique_bytes_removed']==plan['unique_bytes']
    assert result['last_link_allocated_bytes_released']>0
    assert reference(Heat(),tmp_path/'references')['frames']


def test_shared_stage_active_race_stale_plan_and_symlink(tmp_path):
    ae=weights(tmp_path,'ae','autoencoder');a=weights(tmp_path,'flow-a',dependency={'path':str(ae/'coordinate.pt')});weights(tmp_path,'flow-b',dependency={'path':str(ae/'coordinate.pt')})
    p=storage.preview(tmp_path,selected=['flow-a']);assert p['shared_retained'] and {g['id'] for g in p['models']}=={'flow-a'}
    with pytest.raises(ValueError,match='retained'):storage.preview(tmp_path,selected=['ae'])
    with run_lock(a):
        with pytest.raises(ValueError,match='active'):storage.preview(tmp_path,all_models=True)
        with pytest.raises(ValueError,match='owns'):storage.apply(tmp_path,p['id'],p['confirmation'],p['digest'])
    (a/'coordinate.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):storage.apply(tmp_path,p['id'],p['confirmation'],p['digest'])
    outside=tmp_path.parent/'outside-playground-storage';outside.mkdir(exist_ok=True);(outside/'secret').write_text('protected')
    (tmp_path/'escape').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError):storage._inside(tmp_path,'escape/secret')
    with pytest.raises(ValueError):storage._inside(tmp_path,'../escape')
    assert (outside/'secret').read_text()=='protected'


def test_optional_owned_data_logs_scope_and_partial_failure(tmp_path,monkeypatch):
    ae=weights(tmp_path,'ae','autoencoder');flow=weights(tmp_path,'flow',dependency={'path':str(ae/'coordinate.pt')})
    data=tmp_path/'data';data.mkdir();(data/'dataset.npz').write_bytes(b'owned generated data')
    identity=file_digest(data/'dataset.npz');write_json(data/'dataset.json',{'kind':'coordinate_training_data','sha256':identity})
    for p in [*ae.glob('*.pt'),*flow.glob('*.pt')]:
        cp=torch.load(p,weights_only=True);cp['dataset_identity']=identity;torch.save(cp,p)
    (flow/'train.log').write_text('producer log');(flow/'personal.log').write_text('protected unknown log')
    weights_only=storage.preview(tmp_path,all_models=True)
    assert not any(e['path'].endswith(('dataset.npz','train.log')) for e in weights_only['entries'])
    all_scopes=storage.preview(tmp_path,all_models=True,data=True,results=True)
    names={e['path'] for e in all_scopes['entries']}
    assert {'data/dataset.npz','data/dataset.json','flow/train.log'}<=names
    assert 'flow/personal.log' not in names
    unlink=storage._unlink_reviewed
    def fail_one(root,entry):
        if entry['path']=='flow/resume.pt':raise PermissionError('injected disposable failure')
        return unlink(root,entry)
    monkeypatch.setattr(storage,'_unlink_reviewed',fail_one)
    result=storage.apply(tmp_path,all_scopes['id'],all_scopes['confirmation'],all_scopes['digest'])
    assert len(result['failures'])==1 and (flow/'resume.pt').exists()
    assert (flow/'personal.log').exists() and not (data/'dataset.npz').exists()


def test_malicious_lock_symlink_and_changed_parent_cannot_escape(tmp_path):
    owner=weights(tmp_path,'flow')
    outside=tmp_path.parent/'protected-lock-test';outside.write_text('do not touch')
    (owner/'.run.lock').symlink_to(outside)
    with pytest.raises(ValueError,match='active'):storage.preview(tmp_path,all_models=True)
    assert outside.read_text()=='do not touch'


def test_queued_job_protects_refinement_dependency(tmp_path):
    fine=weights(tmp_path,'fine')
    write_json(tmp_path/'jobs/one/job.json',{'status':'queued'})
    write_json(tmp_path/'jobs/one/request.json',{'model_id':'not-this-model','refinement_model_id':file_digest(fine/'coordinate.pt')[:32]})
    with pytest.raises(ValueError,match='active'):storage.preview(tmp_path,all_models=True)


def test_api_isolation_and_compatibility_without_worker_reference(tmp_path):
    from pdeflow.workbench.api import create_app
    from pdeflow.coordinate.registry import register_checkpoint
    from pdeflow.coordinate.contracts import CoordinateConfig
    from pdeflow.coordinate.training import read_checkpoint
    checkpoint=Path('outputs/coordinate-20261027/workspace/coordinate/heat-flow/coordinate.pt')
    if not checkpoint.exists():pytest.skip('Local preserved smoke export unavailable; portable API tested separately')
    entry=register_checkpoint(checkpoint,tmp_path)
    app=create_app(workspace=tmp_path,model_roots=[])
    with TestClient(app) as client:
        r=client.post('/api/playground/reference',json=Heat().model_dump());assert r.status_code==200,r.text
        ref=r.json();s=client.post('/api/playground/measurements',json={'reference_id':ref['id'],'count':25,'time':0}).json()
        base={'model_id':file_digest(checkpoint)[:32],'problem':ref['problem'],'rows':s['rows'],'device':'cpu'}
        assert client.post('/api/coordinate/validate',json=base).status_code==200
        bad={**base,'problem':{**base['problem'],'known_parameters':{'diffusivity':.5}}}
        assert client.post('/api/coordinate/validate',json=bad).status_code==422
        assert client.post('/api/coordinate/reconstruct',json={**base,'initial':ref['initial']}).status_code==422
        # Intercept queue boundary, rather than mock numerical integration.
        captured={}
        def submit(request,checkpoint,token):captured.update(request);return {'id':'a'*32,'status':'queued'}
        app.state.jobs.submit=submit
        assert client.post('/api/coordinate/reconstruct',json=base).status_code==202
        assert set(captured)=={'kind','model_id','problem','rows','queries','inference','runtime','mode'}
        assert not any(k in str(captured.keys()) for k in ('initial','reference','clean'))
        assert not captured['problem']['known_fields']


def test_reimport_repairs_deleted_owned_copy_registration(tmp_path):
    from shutil import copyfile
    from pdeflow.coordinate.registry import register_checkpoint,runs
    source=Path('outputs/coordinate-20261027/workspace/coordinate/heat-flow/coordinate.pt')
    if not source.exists():pytest.skip('Preserved smoke export unavailable')
    first=tmp_path/'first.pt';second=tmp_path/'second.pt'
    copyfile(source,first);copyfile(source,second)
    old=register_checkpoint(first,tmp_path);first.unlink()
    assert not any(r.get('available') for r in runs(tmp_path))
    register_checkpoint(second,tmp_path)
    assert any(r.get('available') and r['export']==str(second) for r in runs(tmp_path))
    assert read_json(tmp_path/'coordinate'/old['id']/'run.json')['previous_missing_export']==str(first)
