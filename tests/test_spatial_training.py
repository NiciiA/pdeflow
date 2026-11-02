"""Real tiny A1/A2/B stages, explicit warm start, strict export and exact resume."""
from dataclasses import replace
import torch
import pytest
from pdeflow.colab import training_preset
from pdeflow.coordinate.experiments import conditional_experiment
from pdeflow.coordinate.training import train,read_checkpoint,portable_checkpoint
from pdeflow.coordinate.model import Velocity
from pdeflow.coordinate.dependency_compatibility import initialize_flow
from pdeflow.coordinate.inference import CoordinateBundle
from pdeflow.runtime import file_digest


def test_four_real_objectives_warm_start_resume_encoder_and_legacy(tmp_path):
    torch.set_num_threads(2)
    c=training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16,'training_options':{'validation_every':20,'broad_every':20}})
    train(c.to_dict(),tmp_path/'autoencoder','autoencoder');train(c.to_dict(),tmp_path/'evolution','evolution',tmp_path/'autoencoder/autoencoder.pt')
    dep=tmp_path/'evolution/evolution.pt';original=file_digest(dep);base=conditional_experiment(c,'B0');train(base.to_dict(),tmp_path/'B0','flow',dep);parent=tmp_path/'B0/coordinate.pt'
    cp,_=read_checkpoint(parent);v0=Velocity(base);v0.load_state_dict(cp['velocity'],strict=True)
    for name in ('B1','B2','B3'):
        cfg=conditional_experiment(c,name,{'warmup_updates':0,'minimum_s':0,'sensor_budget':2});cfg=replace(cfg,training_options={**cfg.training_options,'flow_warm_start':str(parent)}).validate()
        v=Velocity(cfg);audit=initialize_flow(v,cfg,parent,dep,read_checkpoint(dep)[0]);assert audit['copied_keys'] and (bool(audit['new_keys'])==(name!='B2'))
        if name!='B2':assert torch.count_nonzero(v.sensor_attention.out.weight)==0
        from pdeflow.coordinate.model import Autoencoder
        ae=Autoencoder(c).eval();ae.load_state_dict(read_checkpoint(dep)[0]['autoencoder']);f=torch.tensor([0,1]);p=torch.full((2,),.4);known=torch.tensor([True,False]);obs=(torch.rand(2,3,3),torch.ones(2,3,3),torch.ones(2,3,dtype=torch.bool),torch.zeros(2,3,dtype=torch.long));z=torch.randn(2,c.tokens,c.width);flowtime=torch.full((2,),.7);graph=ae.graph_encoder(f)
        torch.testing.assert_close(v0(z,flowtime,p,obs,known,f,graph),v(z,flowtime,p,obs,known,f,graph),atol=2e-6,rtol=2e-5)
        root=tmp_path/name;train(cfg.to_dict(),root,'flow',dep,stop_after=1)
        first=read_checkpoint(root/'resume.pt')[0];assert first['progress']==1 and first['warm_start']['parent_sha256']==file_digest(parent)
        record=train(cfg.to_dict(),root,'flow',resume=True);assert record['progress']==2
        saved,actual=read_checkpoint(root/'resume.pt');assert any(k.startswith('encoder.') for k in saved['autoencoder']) and saved['optimizer'] and saved['ambient_rng']
        assert (root/'flow-preview.json').exists()
        info=saved['history'][-1]['components'];assert info['flow_matching']>0
        if name in ('B2','B3'):assert info['objective_observation_auxiliary']>0 and info['observation_eligible_examples']>0
        assert file_digest(dep)==original
        b=CoordinateBundle(root/'coordinate.pt','cpu');assert b.cp['architecture']==saved['architecture']
        with pytest.raises(ValueError,match='Incompatible'):train(replace(actual,joint={**actual.joint,'conditioning':{'version':'pooled_v1' if name!='B2' else 'spatial_v1'}}).to_dict(),root,'flow',resume=True)
    assert CoordinateBundle(parent,'cpu').velocity.sensor_attention is None
    from pdeflow.colab import profile
    preflight=profile(cfg,tmp_path,True,updates=6,warmup=1,frozen_run=tmp_path)
    assert preflight['success'] and set(preflight['stages'])=={'flow'}
    assert preflight['frozen_stage_reuse']['evolution']==original
    assert file_digest(dep)==original


def test_snapshot_includes_frozen_source_and_streaming_histories(tmp_path):
    from pdeflow import persistence
    local=tmp_path/'local';(local/'source').mkdir(parents=True);(local/'frozen-source').mkdir();(local/'flow').mkdir()
    (local/'source/source.zip').write_bytes(b'new source');(local/'frozen-source/source.zip').write_bytes(b'exact old physics source');(local/'flow/history-live.jsonl').write_text('{"successful_update":1}\n')
    snapshot=persistence.publish(local,tmp_path/'persistent')
    assert 'frozen-source/source.zip' in snapshot['files'] and 'flow/history-live.jsonl' in snapshot['files']
    persistence.restore(tmp_path/'persistent',tmp_path/'restored',snapshot['id'])
    for name in snapshot['files']:assert (tmp_path/'restored'/name).read_bytes()==(local/name).read_bytes()


def test_session_passes_canonical_frozen_a2_to_flow_controller(tmp_path,monkeypatch):
    # Routing-only assertion; real numerical stage execution is the test above.
    from pdeflow.colab import Session
    import pdeflow.training_workflow as workflow
    c=conditional_experiment(training_preset('smoke','cpu','fp32'),'B3')
    c=replace(c,training_options={**c.training_options,'flow_warm_start':'declared-parent.pt'})
    session=Session.new(c,'route-check',tmp_path/'local',tmp_path/'persistent',allow_cpu_check=True,persistence_enabled=False)
    (session.root/'evolution').mkdir();dependency=session.root/'evolution/evolution.pt';dependency.write_bytes(b'non-numerical routing fixture')
    captured={}
    def controller(*args,**kwargs):
        captured['dependency']=args[4];return {'state':'interrupted'}
    monkeypatch.setattr(workflow,'coordinate_stages',controller)
    session.run(stages='flow')
    assert captured['dependency']==dependency
