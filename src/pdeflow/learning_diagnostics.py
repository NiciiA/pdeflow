"""Bounded optimization, prior-distribution and fixed-validation inverse checks."""
from dataclasses import replace
from pathlib import Path
import copy,time
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,write_json,code_identity,file_digest
from pdeflow.training import load_inference
from pdeflow.data.physical import specification,solve_spec,combine
from pdeflow.generative import LinearCFM
from pdeflow.observations import sensor_observations
from pdeflow.sampling import sample
from pdeflow.evaluation import metrics
from pdeflow.specialists import get_specialist
from pdeflow.flow_verification import gaussian_toy


def distribution_summary(fields,dimension):
    """Across-particle statistics, per field/time spectra and RMS; no paired claim."""
    a=np.asarray(fields,dtype=np.float64);axes=tuple(range(a.ndim-dimension,a.ndim))
    spectrum=np.abs(np.fft.fftn(a,axes=axes,norm='ortho'))**2
    freq=np.meshgrid(*[np.fft.fftfreq(n)*n for n in a.shape[-dimension:]],indexing='ij')
    radius=np.floor(np.sqrt(sum(v*v for v in freq))).astype(int)
    radial=[float(spectrum[...,radius==r].mean()) for r in range(min(32,int(radius.max())+1))]
    return {'particles':len(a),'mean_by_field_time':a.mean((0,*axes)).tolist(),'variance_by_field_time':a.var((0,*axes)).tolist(),'RMS_by_field_time':np.sqrt(np.mean(a*a,axis=(0,*axes))).tolist(),'radial_spatial_power':radial,'spectrum_convention':'orthonormal native FFT, radial integer-index shells; averaged over particles/fields/times'}


def run(checkpoint,output,device='cpu',updates=100,cases=2,seeds=(91,92,93)):
    if not 1<=updates<=500 or not 1<=cases<=4 or not 2<=len(seeds)<=4:raise ValueError('Bounded diagnostics require 1–500 updates, 1–4 validation cases and 2–4 fixed seeds')
    bundle=load_inference(checkpoint,RuntimeSpec(device=device));c=bundle.config;g=c.grid
    if np.prod(g.shape)>16384 or g.dimension==3 and max(g.shape)>16:raise ValueError('Diagnostic study is bounded to <=16384 native spatial nodes (3D axes <=16). Use a small compatible model; no silent coarsening.')
    if c.provider is None:raise ValueError('Diagnostics require a physical-provider checkpoint with recorded distribution')
    out=Path(output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    config=replace(c,data=replace(c.data,seed=c.data.seed+8801));physical=[]
    for i in range(cases):
        spec,t=specification(config,'validation',i);physical.append(solve_spec(config,spec,t))
    # Fixed physical set, fixed source/time pairs: optimization test, not online loss.
    train=copy.deepcopy(bundle.model);train.requires_grad_(True);train.train();opt=torch.optim.Adam(train.parameters(),lr=c.training.learning_rate)
    values,conditions=combine(physical);dev=next(train.parameters()).device;dtype=next(train.parameters()).dtype
    u=bundle.layout.pack(bundle.transform.encode(values)).to(device=dev,dtype=dtype);conditions=conditions.to(device=dev,dtype=dtype)
    form=LinearCFM();x,y,s=form.training_state(u,torch.Generator().manual_seed(710),torch.Generator().manual_seed(711))
    before=[v.detach().clone() for v in train.parameters()];history=[]
    with torch.no_grad():initial=float(form.loss(train(x,s,conditions),y))
    fit_start=time.perf_counter()
    for i in range(updates):
        loss=form.loss(train(x,s,conditions),y);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(train.parameters(),10.);opt.step()
        if i%10==0 or i==updates-1:history.append({'update':i+1,'fixed_pair_loss':float(loss.detach())})
    with torch.no_grad():final=float(form.loss(train(x,s,conditions),y))
    overfit={'updates':updates,'initial_loss':initial,'final_loss':final,'history':history,'weights_changed':any(not torch.equal(a,b) for a,b in zip(before,train.parameters())),'parameters':sum(p.numel() for p in train.parameters()),'seconds':time.perf_counter()-fit_start,'meaning':'Fits a frozen physical/source/time pair set starting from the selected export. Memorization/optimization only, not generalization. Original checkpoint never modified.'}
    del train,opt
    reports=[];priors=[];references=[]
    for case,physical_case in enumerate(physical):
        p=physical_case.problem;truth=physical_case.values[None]
        obs=sensor_observations(truth,g,min(10,int(np.prod(g.shape))),77+case,c.inference.noise_std,c.inference.noiseless_tolerance if hasattr(c.inference,'noiseless_tolerance') else .1)
        baselines=get_specialist(c.specialist).baseline(p,obs,c.inference)
        baseline_reports={name:metrics(result,p,obs,truth) for name,result in baselines.items()}
        for seed in seeds:
            for steps in (4,8):
                inf=replace(c.inference,steps=steps,particles=4,seed=seed,controller=None,fk=None,snapshots=None)
                for mode in ('unguided','observation','observation_pde'):
                    result=sample(bundle,p,obs,inf,mode)
                    reports.append({'case':case,'physical_id':physical_case.specification['id'],'seed':seed,'steps':steps,'mode':mode,'settings':__import__('dataclasses').asdict(inf),'metrics':metrics(result,p,obs,truth)})
                    if mode=='unguided' and steps==8:priors.append(result.samples[0].numpy());references.append(np.repeat(truth.numpy(),4,axis=0))
        write_json(out/f'baseline-{case}.json',baseline_reports)
    toy=gaussian_toy(out/'gaussian',updates=1000,particles=2048,steps=48)
    from pdeflow.runtime import process_peak_memory_bytes
    report={'source':code_identity(),'checkpoint':str(Path(checkpoint).resolve()),'checkpoint_sha256':file_digest(checkpoint),'checkpoint_updates':__import__('pdeflow.training',fromlist=['_read_checkpoint'])._read_checkpoint(checkpoint)[0]['progress'],'architecture':{'config':__import__('dataclasses').asdict(c.model),'parameters':sum(p.numel() for p in bundle.model.parameters())},'overfit':overfit,'tractable_distribution':toy,'prior':distribution_summary(np.concatenate(priors),g.dimension),'held_out_reference_distribution':distribution_summary(np.concatenate(references),g.dimension),'reconstruction':reports,'fixed_validation_ids':[p.specification['id'] for p in physical],'seeds':list(seeds),'process_peak_rss_bytes':process_peak_memory_bytes(),'seconds':time.perf_counter()-start,'hypotheses':{'tested':['actual checkpoint budget','fixed-pair optimization','finite-budget tractable flow','conditioned prior statistics','several fixed inverse seeds and integration budgets against conventional baseline'],'not_established':['sufficient long training','capacity as sole cause','calibrated posterior','high-resolution quality']},'interpretation':'Prior distribution statistics differ from paired inverse error. Tiny fixed-pair overfit does not imply generalization; fixed validation is never final test selection.'}
    write_json(out/'diagnostics.json',report);return report
