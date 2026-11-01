"""Scientific objective checks against analytic fields and actual shared modules."""
from dataclasses import replace
import math
import pytest
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.observation_objective import settings,clean_endpoint,endpoint_loss,decode_points
from pdeflow.coordinate.evolution_objective import phase_spectral_mse,decoded_residuals,physics_loss
from pdeflow.coordinate.model import Autoencoder,Velocity
from pdeflow.coordinate.joint import path,active_mask,flow_loss


def tiny(aux=None):
    torch.set_num_threads(2)
    return training_preset('smoke','cpu','fp32',{'encoder_points':16,'query_points':16,'field_loss':{'type':'region_balanced_mse','analysis_resolution':8,'region_queries':16,'time_groups':1},'joint':{'observation_auxiliary':aux or {}}})


def inputs(c,b=2):
    torch.manual_seed(193);ae=Autoencoder(c).eval().requires_grad_(False);flow=Velocity(c).eval();f=torch.arange(b)%2;p=torch.full((b,),.4);known=torch.arange(b)%2==0
    z=torch.randn(b,c.tokens,c.width);source=torch.randn_like(z);s=torch.full((b,),.8);state,target,_=path(c,z,source,s,known,f)
    q=torch.rand(b,4,3);feat=torch.randn(b,4,3);feat[...,1]=.2;feat[...,2]=1;mask=torch.ones(b,4,dtype=torch.bool);ids=torch.zeros(b,4,dtype=torch.long)
    ids[f.eq(1),1::2]=1;obs=(q,feat,mask,ids)
    norms={'heat':[{'mean':.3,'scale':2.}], 'wave':[{'mean':-.2,'scale':3.},{'mean':.4,'scale':5.}]}
    return ae,flow,f,p,known,state,target,s,obs,norms


def evaluate(c,args,pred,obs=None,seed=3):
    ae,flow,f,p,known,state,target,s,o,norms=args
    return endpoint_loss(ae,c,state,pred,s,torch.zeros(1,c.tokens,c.width,device=state.device),torch.ones(1,c.tokens,c.width,device=state.device),p,known,f,obs or o,norms,0,seed)


def test_zero_beta_exact_no_decode_and_endpoint_inverse_once(monkeypatch):
    c=tiny();a=inputs(c);pred=a[1](a[5],a[7],a[3],a[8],a[4],a[2],a[0].graph_encoder(a[2]));base=flow_loss(c,pred,a[6],a[4],a[2])
    def fail(*args,**kwargs):raise AssertionError('zero beta decoded')
    monkeypatch.setattr(a[0],'evolve',fail);extra,_=evaluate(c,a,pred);torch.testing.assert_close(base+extra.mean(),base,atol=0,rtol=0)
    x=torch.randn(2,3,4);v=torch.randn_like(x);s=torch.tensor([.2,.9]);m=torch.ones(1,3,4)*2;scale=m*3
    torch.testing.assert_close(clean_endpoint(x,v,s,m,scale),(x+(1-s[:,None,None])*v)*scale+m)


def test_real_sensor_path_gaussian_units_masks_and_frozen_gradients():
    c=tiny({'beta':.03,'minimum_s':0,'sensor_budget':4,'gaussian_constant':True});a=inputs(c);ae,flow,f,p,known,state,target,s,obs,norms=a
    before={k:v.clone() for k,v in ae.state_dict().items()};pred=flow(state,s,p,obs,known,f,ae.graph_encoder(f))
    extra,info=evaluate(c,a,pred);endpoint=clean_endpoint(state,pred,s,torch.zeros_like(state[:1]),torch.ones_like(state[:1]))
    # Independent old query-by-query decoder at exact physical x/y/t and head.
    rows=[]
    for b in range(2):
        penalties=[]
        for j in range(4):
            kind=int(obs[3][b,j]);n=norms[('heat','wave')[int(f[b])]][kind]
            value=ae.decode(endpoint[b:b+1],obs[0][b,j:j+1],p[b:b+1],known[b:b+1],('u','v')[kind],f[b:b+1])[0,0]
            sigma=obs[1][b,j,1]*n['scale'];r=(value-obs[1][b,j,0])*n['scale']
            penalties.append(.5*(r/sigma).square()+.5*torch.log(2*math.pi*sigma.square()))
        rows.append(torch.stack(penalties).mean())
    torch.testing.assert_close(extra,.03*torch.stack(rows),atol=2e-6,rtol=3e-6)
    extra.mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in flow.observation_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in flow.processor.parameters())
    assert all(p.grad is None for p in ae.parameters())
    for k,v in ae.state_dict().items():assert torch.equal(v,before[k])
    assert info['observation_gaussian_values']==8
    # Padded NaNs are ignored, an empty case contributes zero (not invented data).
    q,feat,valid,ids=[x.clone() for x in obs];valid[0]=False;q[0]=float('nan');feat[0]=float('nan');ids[0]=99
    value,_=evaluate(c,a,pred,(q,feat,valid,ids));assert value[0]==0 and torch.isfinite(value).all()


def test_noiseless_explicit_tolerance_and_duplicate_rejection():
    c=tiny({'beta':1.,'minimum_s':0,'sensor_budget':4,'tolerance_absolute':.1,'tolerance_relative':0.});a=inputs(c)
    q,feat,valid,ids=[x.clone() for x in a[8]];feat[...,1]=0
    pred=torch.zeros_like(a[5],requires_grad=True);value,info=evaluate(c,a,pred,(q,feat,valid,ids));assert info['observation_tolerance_values']==8 and torch.isfinite(value).all()
    rejected=replace(c,joint={**c.joint,'observation_auxiliary':{**c.joint['observation_auxiliary'],'noiseless_policy':'reject'}}).validate()
    with pytest.raises(ValueError,match='Zero-noise'):evaluate(rejected,a,pred,(q,feat,valid,ids))
    q[:,1]=q[:,0];ids[:,1]=ids[:,0]
    with pytest.raises(ValueError,match='Duplicate'):evaluate(c,a,pred,(q,feat,valid,ids))


def test_unknown_scalar_has_no_hidden_parameter_leak_and_known_authority():
    c=tiny({'beta':.01,'minimum_s':0,'sensor_budget':4});a=inputs(c);pred=torch.zeros_like(a[5]);one,_=evaluate(c,a,pred)
    changed=list(a);changed[3]=a[3].clone();changed[3][1]=.99;two,_=evaluate(c,changed,pred)
    torch.testing.assert_close(one[1],two[1],atol=0,rtol=0)
    torch.testing.assert_close(a[0].decode_parameter(a[5],a[3],a[4],False,a[2])[0],a[3][0],atol=0,rtol=0)
    changed[3][0]=.9;three,_=evaluate(c,changed,pred);assert not torch.isclose(one[0],three[0])


def test_microbatch_incomplete_reduction_and_subsample_gradient_equivalence():
    c=tiny({'beta':.01,'minimum_s':0,'sensor_budget':2});a=inputs(c,5);pred=torch.randn_like(a[5],requires_grad=True)
    full,_=evaluate(c,a,pred);gradient=torch.autograd.grad(full.mean(),pred)[0];pieces=[]
    for start,end in ((0,3),(3,5)):
        ae,flow,f,p,k,state,target,s,obs,norms=a
        part=(ae,flow,f[start:end],p[start:end],k[start:end],state[start:end],target[start:end],s[start:end],tuple(x[start:end] for x in obs),norms)
        loss,_=evaluate(c,part,pred[start:end],seed=3+start);pieces.append(loss)
    other=torch.cat(pieces);torch.testing.assert_close(full,other,atol=2e-6,rtol=3e-6)
    torch.testing.assert_close(gradient,torch.autograd.grad(other.mean(),pred)[0],atol=2e-6,rtol=3e-6)


def test_flow_activation_range_warmup_and_noise_validation():
    c=tiny({'beta':.2,'minimum_s':.9});a=inputs(c);v=torch.zeros_like(a[5]);value,_=evaluate(c,a,v);assert value.eq(0).all()
    for bad in ({'beta':-1},{'covariance':'dense'},{'noiseless_policy':'floor'},{'minimum_s':1.1}):
        with pytest.raises(ValueError):settings(bad)


def test_phase_sensitive_complex_spectral_parseval():
    n=16;x=torch.arange(n)*2*math.pi/n;u=torch.cos(x)[:,None].expand(n,n);v=torch.sin(x)[:,None].expand(n,n).clone().requires_grad_()
    # Equal power spectra, different phases: a magnitude-only loss would be zero.
    assert torch.allclose(torch.fft.fft2(u).abs(),torch.fft.fft2(v).abs(),atol=3e-5)
    loss=phase_spectral_mse(v,u);torch.testing.assert_close(loss,(v-u).square().mean(),atol=2e-6,rtol=2e-6)
    assert loss>.9;loss.backward();assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0
    assert phase_spectral_mse(u,u)==0


def test_real_evolution_residual_has_time_space_gradient_path_and_periodic_derivatives():
    c=tiny();a=inputs(c);ae=a[0];ae.evolution.requires_grad_(True);ae.graph_encoder.requires_grad_(True)
    q=torch.tensor([[[.13,.27,.237],[.2,.3,.6]],[[.31,.12,.237],[.4,.8,.6]]]);r=decoded_residuals(ae,a[5],q,a[3],a[4],a[2],a[9]);sum(x.square().mean() for x in r.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 and torch.isfinite(p.grad).all() for p in ae.evolution.parameters())
    q=torch.tensor([[[0.,.2,.237],[1.,.2,.237]]],requires_grad=True);z=a[5][1:];f=torch.ones(1,dtype=torch.long);u=decode_points(ae,z,q,torch.zeros(1,2,dtype=torch.long),a[3][1:],a[4][1:],f)
    du=torch.autograd.grad(u.sum(),q,create_graph=True)[0];torch.testing.assert_close(u[:,0],u[:,1],atol=2e-6,rtol=2e-5);torch.testing.assert_close(du[:,0,:2],du[:,1,:2],atol=2e-5,rtol=2e-4)


class Manufactured:
    """Independent differentiable periodic heat/wave fields, no learned operator."""
    c=tiny()
    def graph_encoder(self,f):return torch.zeros(len(f),1)
    def evolve(self,z,ta,tb,p,known,f,graph):return torch.stack([tb,p,f.float()],-1)[:,None]
    def decode_parameter(self,z,p,known,physical,f):return p
    def decode_state(self,z,xy,quantity,f):
        t=z[:,0,0,None];p=z[:,0,1,None];a=torch.cos(2*math.pi*xy[...,0]);omega=2*math.pi*p
        if quantity==0:return torch.where(f[:,None].eq(0),a*torch.exp(-p*(2*math.pi)**2*t),a*torch.cos(omega*t)+2*t)
        return torch.where(f[:,None].eq(1),-a*omega*torch.sin(omega*t)+2.,a*0)


def test_independent_manufactured_heat_wave_normalized_residuals():
    torch.manual_seed(3);q=torch.rand(2,5,3);f=torch.tensor([0,1]);p=torch.tensor([.07,.4]);z=torch.zeros(2,1,3);norms={'heat':[{'mean':0.,'scale':1.}],'wave':[{'mean':0.,'scale':1.},{'mean':0.,'scale':1.}]}
    r=decoded_residuals(Manufactured(),z,q,p,torch.ones(2,dtype=torch.bool),f,norms)
    assert r['heat'][0].abs().max()<2e-6 and r['wave_u'][1].abs().max()<2e-6 and r['wave_v'][1].abs().max()<3e-6


def test_public_residual_report_does_not_sum_derivatives_across_particles():
    from pdeflow.coordinate.physics import strong_residual
    c=tiny();a=inputs(c);ae=a[0];z=a[5][1:].repeat(2,1,1);z[1]+=1.;f=torch.ones(2,dtype=torch.long);p=torch.tensor([.3,.6]);k=torch.ones(2,dtype=torch.bool);q=torch.tensor([[.2,.3,.237]])
    both=strong_residual(ae,z,q,p,k,f,a[9]['wave'])
    for i in range(2):
        one=strong_residual(ae,z[i:i+1],q,p[i:i+1],k[i:i+1],f[i:i+1],a[9]['wave'])
        for name in both:torch.testing.assert_close(both[name][i],one[name][0],atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('kind',['mse','region_balanced_mse','spectral_mse'])
def test_actual_a1_a2_selected_field_objectives(tmp_path,kind):
    from pdeflow.coordinate.training import train,read_checkpoint
    c=tiny();loss={**c.field_loss,'type':kind,'spectral_weight':.1 if kind=='spectral_mse' else 0.,'spectral_resolution':8}
    c=replace(c,field_loss=loss).validate();a1=train(c.to_dict(),tmp_path/'a1','autoencoder');a2=train(c.to_dict(),tmp_path/'a2','evolution',a1['export'])
    for record in (a1,a2):
        cp=read_checkpoint(record['export'])[0]
        assert cp['field_loss']['type']==kind and cp['history'][0]['gradients_finite']
        if kind=='spectral_mse':assert cp['history'][0]['components']['spectral_field']>0


def test_actual_b_auxiliary_batched_matches_accumulated_gradients(tmp_path,monkeypatch):
    from test_batched_training import compare,config
    from pdeflow.coordinate.training import train
    from pathlib import Path
    c=config();c=replace(c,joint={**c.joint,'observation_auxiliary':{'beta':.0001,'minimum_s':0.,'sensor_budget':2,'discrepancy_relative':.05}}).validate()
    a=train(c.to_dict(),tmp_path/'a1','autoencoder');b=train(c.to_dict(),tmp_path/'a2','evolution',a['export'])
    compare(tmp_path,monkeypatch,c,'flow',Path(b['export']))


def test_budget_conversion_requires_explicit_compatible_warmup():
    from pdeflow.training_settings import configure
    c=replace(tiny(),ae_steps=10000,evolution_steps=10000,flow_steps=10000,batch_size=8,
        training_options={**tiny().training_options,'autoencoder':{'warmup':1000},'flow':{'warmup':1000}}).validate()
    with pytest.raises(ValueError,match='warmup=1000'):configure(c,{'effective_batch':256,'microbatch':8},'keep_examples')
    updated,report=configure(c,{'effective_batch':256,'microbatch':8,'schedule_policy':'rescale_warmup'},'keep_examples')
    assert updated.ae_steps==313 and updated.training_options['autoencoder']['warmup']==31
    assert report['rounding']['autoencoder']['warmup_change']['policy']=='rescale_warmup'


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Actual MPS unavailable')
def test_actual_mps_endpoint_and_spectral_gradients():
    c=tiny({'beta':.001,'minimum_s':0.,'sensor_budget':2,'discrepancy_relative':.05});a=list(inputs(c));ae,flow=a[:2];ae.to('mps');flow.to('mps')
    for i in range(2,8):a[i]=a[i].to('mps')
    a[8]=tuple(x.to('mps') for x in a[8]);pred=flow(a[5],a[7],a[3],a[8],a[4],a[2],ae.graph_encoder(a[2]));loss,_=evaluate(c,a,pred);loss.mean().backward();torch.mps.synchronize()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in flow.observation_encoder.parameters())
    x=torch.rand(8,8,device='mps',requires_grad=True);target=torch.rand(8,8,device='mps');value=phase_spectral_mse(x,target);value.backward();assert torch.isfinite(x.grad).all()
    ae.evolution.requires_grad_(True);ae.graph_encoder.requires_grad_(True)
    r=decoded_residuals(ae,a[5],a[8][0][:,:1],a[3],a[4],a[2],a[9]);sum(v.square().mean() for v in r.values()).backward();torch.mps.synchronize()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in ae.evolution.parameters())


def test_a2_physics_ablation_reuses_a1_but_rejects_same_stage_objective_change(tmp_path):
    from pdeflow.coordinate.optimization import representation_contract,model_contract,resume_contract
    from pdeflow.coordinate.training import train,read_checkpoint
    c=tiny();changed=replace(c,evolution={**c.evolution,'physics':{'weight':.0001,'query_points':1}}).validate()
    assert representation_contract(c)==representation_contract(changed)
    assert model_contract(c)!=model_contract(changed)
    a1=train(c.to_dict(),tmp_path/'a1','autoencoder')
    a2=train(changed.to_dict(),tmp_path/'a2','evolution',a1['export'])
    cp=read_checkpoint(tmp_path/'a2/resume.pt')[0]
    with pytest.raises(ValueError,match='Incompatible model/data'):resume_contract(cp,c,'evolution')
    assert a2['progress']==2
