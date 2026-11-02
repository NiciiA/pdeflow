"""Physical generated variables: nodal TRAIN-normalized u0/v0, normalized log scalar."""
import torch


def parameter_normalize(c,value,families):
    r=c.reference_config();bounds=value.new_tensor([r.diffusivity_range,r.wave_speed_range])[families.long()]
    return 2*torch.log(value/bounds[:,0])/torch.log(bounds[:,1]/bounds[:,0])-1


def parameter_physical(c,value,families):
    r=c.reference_config();bounds=value.new_tensor([r.diffusivity_range,r.wave_speed_range])[families.long()]
    result=bounds[:,0]*torch.exp((value+1)*.5*torch.log(bounds[:,1]/bounds[:,0]))
    if not torch.isfinite(result).all():raise FloatingPointError('Generated log coefficient overflowed; no silent clipping')
    return result


def active_mask(c,known,families):
    mask=torch.ones(len(known),c.nodes+1,2,dtype=torch.bool,device=known.device);mask[:,:c.nodes,1]=families[:,None].eq(1);mask[:,-1,0]=~known;mask[:,-1,1]=False
    return mask


def enforce(c,state,parameter,known,families):
    mask=active_mask(c,known,families);out=state*mask.to(state.dtype);out=out.clone();out[:,-1,0]=torch.where(known,parameter,out[:,-1,0]);return out


def pack(values,parameter):
    scalar=torch.stack([parameter,torch.zeros_like(parameter)],-1)[:,None];return torch.cat([values,scalar],1)


def group_loss(c,pred,target,known,families):
    """Per-case equal active u/v/scalar group means, then external case average."""
    mse=(pred.float()-target.float()).square();u=mse[:,:c.nodes,0].mean(1);v=mse[:,:c.nodes,1].mean(1);p=mse[:,-1,0]
    return (u+v*families.eq(1)+p*(~known))/(1+families.eq(1).float()+(~known).float())
