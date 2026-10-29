"""Bottleneck reconstruction diagnostics use held-out reference context explicitly."""
from pathlib import Path
import numpy as np
import torch
from pdeflow.runtime import RuntimeSpec,read_json,write_json,file_digest,code_identity
from .training import read_checkpoint,data_config,select_context,normalized_parameter
from .model import Autoencoder,decode_chunks
from .evaluation import error_summary,common_grid_diagnostics

def evaluate(checkpoint,output,device='cpu',split='validation'):
    from .validation import representation
    cp,c=read_checkpoint(checkpoint,'autoencoder')
    return representation(checkpoint,output,device,split,subset=min(c.validation_count if split=='validation' else c.test_count,24))

def _legacy_evaluate(checkpoint,output,device='cpu',split='validation'):
    cp,c=read_checkpoint(checkpoint,'autoencoder')
    if not cp['completed'] or split not in ('validation','test'):raise ValueError('Requires a completed autoencoder and validation/test split')
    root=Path(checkpoint).parent;metadata=read_json(root/'dataset.json')
    if metadata['sha256']!=file_digest(root/'dataset.npz'):raise ValueError('Evaluation data identity changed')
    runtime=RuntimeSpec(device=device);dev,_=runtime.resolve();ae=Autoencoder(c).to(dev).eval().requires_grad_(False)
    ae.load_state_dict(cp['autoencoder']);n=cp['normalization'];grid=data_config(c).grid;records=[]
    with np.load(root/'dataset.npz',allow_pickle=False) as f:values=torch.from_numpy(f['values'].copy());q=torch.from_numpy(f['queries'].copy());d=torch.from_numpy(f['diffusivity'].copy())
    with torch.no_grad():
        for i,(identifier,kind) in enumerate(zip(metadata['ids'],metadata['splits'])):
            if kind!=split:continue
            cq,cv=select_context(q,(values[i:i+1]-n['mean'])/n['scale'],c);p=normalized_parameter(c,d[i:i+1]).to(dev)
            latent=ae.encode(cq[None].to(dev),cv.to(dev),p);prediction=(decode_chunks(ae,latent,q.to(dev),p)*n['scale']+n['mean']).cpu().numpy()
            truth=values[i].numpy();shape=(len(c.times),*c.reference_shape);a=prediction.reshape(shape);b=truth.reshape(shape);axes=tuple(range(1,a.ndim))
            spectrum_a=np.abs(np.fft.rfftn(a,axes=axes))**2;spectrum_b=np.abs(np.fft.rfftn(b,axes=axes))**2
            records.append({'id':identifier,**error_summary(prediction,truth),**common_grid_diagnostics(prediction,grid,float(d[i])),'reference_physics':common_grid_diagnostics(truth[None],grid,float(d[i])),'spectral_power_relative_l2':float(np.linalg.norm(spectrum_a-spectrum_b)/max(np.linalg.norm(spectrum_b),1e-12)),'initial_rmse':float(np.sqrt(np.mean((a[0]-b[0])**2)))})
    report={'stage':'autoencoder','checkpoint_sha256':file_digest(checkpoint),'split':split,'source':code_identity(),'dataset_identity':metadata['sha256'],'records':records,'meaning':'Held-out bottleneck reconstruction with complete reference context supplied to the reference encoder. Not observation-only inference or generalization of the latent flow. Physical/spectral distortion reported separately.'}
    if Path(output).exists():raise FileExistsError('AE evaluation exists; choose a fresh output')
    write_json(output,report);return report
