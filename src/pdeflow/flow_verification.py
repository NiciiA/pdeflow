"""Bounded learned Gaussian example using the production CFM objective/steppers."""
from pathlib import Path
import time
import torch
from torch import nn
from pdeflow.generative import LinearCFM
from pdeflow.sampling import Heun
from pdeflow.runtime import write_json,code_identity

def gaussian_toy(output,updates=2000,particles=8192,steps=64):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.manual_seed(81);torch.set_num_threads(2);start=time.perf_counter()
    model=nn.Sequential(nn.Linear(2,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1));opt=torch.optim.Adam(model.parameters(),lr=.002)
    source=torch.Generator().manual_seed(82);times=torch.Generator().manual_seed(83);targets=torch.Generator().manual_seed(84);form=LinearCFM();mu=1.25;std=.65;before=[p.detach().clone() for p in model.parameters()];losses=[]
    for i in range(updates):
        u=mu+std*torch.randn(256,1,generator=targets);x,y,s=form.training_state(u,source,times);v=model(torch.cat((x,s[:,None]),1));loss=form.loss(v,y);opt.zero_grad();loss.backward();opt.step()
        if i%100==0 or i==updates-1:losses.append({'step':i+1,'training_loss':float(loss.detach())})
    x=torch.randn(particles,1,generator=torch.Generator().manual_seed(85));initial=x.clone();stepper=Heun()
    with torch.no_grad():
        for j in range(steps):x=stepper.advance(lambda a,s:model(torch.cat((a,torch.full_like(a,s)),1)),x,j/steps,1/steps).state
    report={'source':code_identity(),'formulation':form.name,'target':{'mean':mu,'std':std},'generation':{'particles':particles,'steps':steps,'stepper':'production Heun','source_mean':float(initial.mean()),'source_std':float(initial.std()),'mean':float(x.mean()),'std':float(x.std()),'mean_error':abs(float(x.mean())-mu),'std_error':abs(float(x.std())-std),'mean_Monte_Carlo_standard_error':std/particles**.5},'actual_optimizer_updates':updates,'weights_changed':any(not torch.equal(a,b) for a,b in zip(before,model.parameters())),'held_out_target_stream':'independent analytic target, no test-based selection','history':losses,'seconds':time.perf_counter()-start,'interpretation':'Learned finite-budget approximation; Monte Carlo and learning/integration error are separate. Not exact Gaussian moments or PDE validation.'}
    write_json(output/'toy.json',report);torch.save({'model':model.state_dict(),'samples':x,'initial':initial},output/'toy.pt');return report
