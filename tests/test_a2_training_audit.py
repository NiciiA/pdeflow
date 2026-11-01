"""Real frozen-path gradients, current previews and fixed validation; tiny budgets."""
from dataclasses import replace
from pathlib import Path
import json,copy,hashlib,zipfile
import numpy as np
import pytest
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.training import train,read_checkpoint
from pdeflow.coordinate.model import Autoencoder
from pdeflow.coordinate.dataset import example,Dataset
from pdeflow.coordinate.evolution_monitor import evaluate,publish_preview,read_preview,evaluation_guard
from pdeflow.coordinate.evolution_audit import diagnose
from pdeflow.notebook import full_history
from pdeflow.runtime import write_json,file_digest


def config():
    torch.set_num_threads(2)
    return training_preset('pipeline-check','cpu','fp32',{'encoder_points':16,'query_points':16,'field_loss':{'type':'mse'},'checkpoint_every':1,'training_options':{'validation_every':1,'validation_subset':2}})


def test_a2_optimizer_real_changes_frozen_weights_and_exact_component_reduction(tmp_path):
    c=config();a=train(c.to_dict(),tmp_path/'a1','autoencoder');initial=read_checkpoint(a['export'])[0]['autoencoder']
    result=train(c.to_dict(),tmp_path/'a2','evolution',a['export']);cp=read_checkpoint(result['export'])[0]
    assert result['a2_path_audit']['optimizer_ownership_verified']
    for key,val in cp['autoencoder'].items():
        if key.startswith(('encoder.','decoder.','coefficient.')):assert torch.equal(val,initial[key]),key
    assert any(not torch.equal(v,initial[k]) for k,v in cp['autoencoder'].items() if k.startswith('evolution.'))
    for row in cp['history']:
        assert sum(v for k,v in row['components'].items() if k.startswith('objective_'))==pytest.approx(row['training_loss'],abs=1e-6,rel=2e-6)
    probe=cp['history'][0]['a2_update_probe']['modules']
    assert probe['evolution']['parameter_delta_l2']>0 and probe['evolution']['sampled_clipped_gradient_l2']>0
    meta,frames=read_preview(tmp_path/'a2');assert meta['checkpoint_update']==meta['preview_generation_update']==2
    assert {(r['family'],r['quantity']) for r in meta['rows']}=={('heat','u'),('wave','u'),('wave','v')}
    assert all(r['physical_time']>0 for r in meta['rows']) and frames.shape==(3,3,12,12)
    np.testing.assert_allclose(frames[:,2],abs(frames[:,1]-frames[:,0]),atol=1e-7)


def test_fixed_validation_queries_when_optimizer_changes_round_to_zero(tmp_path):
    c=replace(config(),learning_rate=1e-30)
    a=train(c.to_dict(),tmp_path/'a1','autoencoder');b=train(c.to_dict(),tmp_path/'a2','evolution',a['export']);cp=read_checkpoint(b['export'])[0]
    assert cp['history'][0]['validation_loss']==cp['history'][1]['validation_loss']
    assert cp['history'][0]['a2_update_probe']['modules']['evolution']['parameter_delta_l2']==0
    assert 'fixed' in cp['history'][0]['validation_policy']


def test_preview_has_no_training_rng_mode_parameter_or_gradient_side_effects(tmp_path):
    c=config();data=Dataset(c,tmp_path/'data',True);model=Autoencoder(c);model.encoder.eval();model.requires_grad_(False);model.evolution.requires_grad_(True)
    rng=torch.get_rng_state().clone();modes=[m.training for m in model.modules()];weights={k:v.clone() for k,v in model.state_dict().items()}
    publish_preview(tmp_path,c,model,data.metadata['normalizations'],[data.get('validation',i) for i in (0,1)],13)
    assert torch.equal(torch.get_rng_state(),rng) and [m.training for m in model.modules()]==modes
    assert all(p.grad is None for p in model.parameters())
    for k,v in model.state_dict().items():assert torch.equal(weights[k],v),k
    meta,frames=read_preview(tmp_path);assert meta['checkpoint_update']==13
    # Corrupt/incomplete array publication must not show mismatched labels.
    (tmp_path/'evolution-preview.npz').write_bytes(b'partial');assert read_preview(tmp_path) is None


def test_history_merges_full_journal_validation_marker_and_recent_tail(tmp_path):
    rows=[{'step':i,'attempted_update':i,'training_loss':1.,**({'validation_loss':.5,'validation_policy':'fixed independent RNG'} if i==500 else {})} for i in range(1,1301)]
    write_json(tmp_path/'history.json',rows[:1000]);(tmp_path/'history-live.jsonl').write_text('\n'.join(json.dumps(r) for r in rows[1000:])+'\n{partial')
    result=full_history(tmp_path,{'history':rows[-200:],'attempted_updates':1300});assert len(result)==1300
    assert next(r for r in result if r['step']==500)['validation_loss']==.5
    from pdeflow.notebook import history_figure
    fig=history_figure(result);line=next(x for x in fig.axes[0].lines if x.get_label()=='fixed validation objective');assert line.get_marker()=='o' and len(line.get_xdata())==1
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_diagnostic_export_standalone_old_kernel_and_baseline_report(tmp_path):
    c=config();a=train(c.to_dict(),tmp_path/'autoencoder','autoencoder');b=train(c.to_dict(),tmp_path/'evolution','evolution',a['export'])
    before=file_digest(tmp_path/'evolution/resume.pt')
    from pdeflow.evolution_export import export_a2_diagnostics
    result=export_a2_diagnostics(tmp_path,tmp_path/'diagnostic.zip');assert result['contains_diagnostic_weights'] and result['not_a_resume_bundle']
    assert file_digest(tmp_path/'evolution/resume.pt')==before
    with zipfile.ZipFile(tmp_path/'diagnostic.zip') as z:
        z.extract('a2-diagnostic.pt',tmp_path/'exported');summary=json.loads(z.read('diagnostic-summary.json'))
    assert summary['checkpoint']['progress']==2
    exported=tmp_path/'exported/a2-diagnostic.pt';cp,_=read_checkpoint(exported)
    assert 'optimizer' not in cp and any(k.startswith('encoder.') for k in cp['autoencoder'])
    report=diagnose(exported,tmp_path/'report.json',resolution=4)
    rows=report['comparisons'][0]['records'];assert {r['family'] for r in rows}=={'heat','wave'}
    for row in rows:
        for metric in row['per_time']:
            assert metric['learned']['rmse']**2==pytest.approx(metric['learned']['mse'])
            if metric['physical_time']==0:assert metric['learned']['mse']==pytest.approx(metric['held_initial']['mse'])
            assert 'reencoded_reference' in metric
        assert row['normalized_strong_residual_mse']


def test_spectral_power_zero_has_identical_grid_mse_gradient():
    from pdeflow.coordinate.evolution_objective import phase_spectral_mse
    x=torch.randn(2,16,16,requires_grad=True);y=torch.randn_like(x)
    a=phase_spectral_mse(x,y,0.).mean();b=(x-y).square().mean()
    torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(torch.autograd.grad(a,x,retain_graph=True)[0],torch.autograd.grad(b,x)[0],atol=2e-6,rtol=2e-6)


def test_split_spectral_gradient_is_real_and_not_overwritten_by_physics_zero(tmp_path):
    from pdeflow.coordinate import batching
    from pdeflow.coordinate.frozen import FrozenTargets
    from pdeflow.coordinate.profiling import Timings
    from pdeflow.coordinate.dataset import item_quantity
    from pdeflow.field_losses import grouped_queries,differentiable_loss
    c=replace(config(),field_loss={'type':'spectral_mse','spectral_weight':.1,'spectral_resolution':8})
    data=Dataset(c,tmp_path/'data',True);norms=data.metadata['normalizations'];ae=Autoencoder(c).requires_grad_(False);ae.evolution.requires_grad_(True);ae.graph_encoder.requires_grad_(True)
    timings=Timings(torch.device('cpu'),False);cache=FrozenTargets('test',8,timings,None,0);g=torch.Generator().manual_seed(8)
    # Query a deliberately nonzero time for both families.
    rows=[]
    for i in (0,1):
        item=data.get('train',i);queries=[]
        for j,name in enumerate(('u','v') if item['family']=='wave' else ('u',)):
            q,y,w,_=grouped_queries(c,[item_quantity(item,name)],g,norms[item['family']][j],None,[.237]);n=norms[item['family']][j];queries.append((q[0,:,:2],(y[0]-n['mean'])/n['scale'],w[0]))
        rows.append(batching.StatePlan(item,0.,.237,.1185,queries))
    loss,parts=batching.forward(c,ae,rows,'evolution',norms,torch.device('cpu'),timings,cache,differentiable_loss,item_quantity,{},True)
    torch.testing.assert_close(loss,sum(parts.values()),atol=1e-6,rtol=1e-6)
    grads=torch.autograd.grad(parts['spectral'].sum(),list(ae.evolution.parameters()),allow_unused=True)
    assert any(g is not None and g.abs().sum()>0 for g in grads);cache.close()


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual MPS unavailable')
def test_actual_mps_a2_preview_and_physical_baseline(tmp_path):
    c=replace(config(),runtime={'device':'mps','threads':2});a=train(c.to_dict(),tmp_path/'a1','autoencoder');b=train(c.to_dict(),tmp_path/'a2','evolution',a['export'])
    assert read_preview(tmp_path/'a2')[0]['checkpoint_update']==2
    r=diagnose(b['export'],tmp_path/'mps-audit.json',device='mps',resolution=4)
    assert r['evaluation_device']=='mps' and all(x['initial_identity_max_error']==0 for x in r['comparisons'][0]['records'])


def test_optional_nonzero_a2_policy_is_explicit_and_preserves_default_rng():
    from pdeflow.coordinate.temporal import evolution_time,state_time
    from pdeflow.coordinate.optimization import model_contract,representation_contract
    c=config();g=torch.Generator().manual_seed(93);h=torch.Generator().manual_seed(93)
    old=[evolution_time(c,i,g) for i in range(1024)];expected=[state_time(c,i,h) for i in range(1024)]
    assert old==expected and torch.equal(g.get_state(),h.get_state()) and sum(t==0 for t in old)>150
    changed=replace(c,evolution={**c.evolution,'time_policy':'nonzero_conditioned_v1'}).validate()
    g=torch.Generator().manual_seed(93);new=[evolution_time(changed,i,g) for i in range(1024)]
    assert all(0<t<=1 for t in new) and len(set(new))>800
    assert representation_contract(c)==representation_contract(changed) and model_contract(c)!=model_contract(changed)
    # This option must not alter A1's sampler or default c.time_sampling.
    assert state_time(changed,3,torch.Generator().manual_seed(51),'encoder')==state_time(c,3,torch.Generator().manual_seed(51),'encoder')


def test_structural_initial_identity_has_zero_evolution_gradient_but_nonzero_time_does_not(tmp_path):
    from pdeflow.coordinate import batching
    from pdeflow.coordinate.frozen import FrozenTargets
    from pdeflow.coordinate.profiling import Timings
    from pdeflow.coordinate.dataset import item_quantity
    from pdeflow.field_losses import grouped_queries,differentiable_loss
    c=config();data=Dataset(c,tmp_path/'data',True);norms=data.metadata['normalizations'];ae=Autoencoder(c).requires_grad_(False);ae.evolution.requires_grad_(True);ae.graph_encoder.requires_grad_(True);timings=Timings(torch.device('cpu'),False);cache=FrozenTargets('identity',8,timings,None,0)
    norms_at=[]
    for t in (0.,.237):
        rows=[]
        for i in (0,1):
            item=data.get('train',i);qset=[]
            for j,name in enumerate(('u','v') if item['family']=='wave' else ('u',)):
                q,y,w,_=grouped_queries(c,[item_quantity(item,name)],torch.Generator().manual_seed(5),norms[item['family']][j],None,[t]);n=norms[item['family']][j];qset.append((q[0,:,:2],(y[0]-n['mean'])/n['scale'],w[0]))
            rows.append(batching.StatePlan(item,0.,t,t/2,qset))
        ae.zero_grad(set_to_none=True);loss=batching.forward(c,ae,rows,'evolution',norms,torch.device('cpu'),timings,cache,differentiable_loss,item_quantity,{}).mean();loss.backward()
        norms_at.append(sum(float(p.grad.abs().sum()) for p in ae.evolution.parameters() if p.grad is not None))
    cache.close();assert norms_at[0]==0 and norms_at[1]>0
