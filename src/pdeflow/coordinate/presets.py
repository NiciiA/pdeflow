"""Architecture capacities; no device-fit or accuracy claims."""
from dataclasses import replace
import torch
from .joint import schema
LADDER={
 'smoke':(24,1,4,8,1,1,4),
 'xsmall':(256,6,8,64,2,2,16),
 'small':(384,8,8,128,3,2,24),
 'medium':(512,12,8,256,4,3,32),
 'large':(768,16,12,384,4,3,48),
 'xlarge':(1024,24,16,512,6,4,64),
}
def capacity(c,name):
    name={'pipeline-check':'smoke','extra-large':'xlarge'}.get(name.lower(),name.lower())
    if name=='custom':return c.validate()
    if name not in LADDER:raise ValueError('Model capacity: '+', '.join([*LADDER,'custom']))
    w,d,h,t,e,dec,obs=LADDER[name]
    return replace(c,width=w,layers=d,heads=h,tokens=t,joint={**c.joint,'encoder_depth':e,'decoder_depth':dec,'observation_tokens':obs}).validate()

def counts(c):
    from .model import Autoencoder,Velocity
    with torch.device('meta'):ae=Autoencoder(c);v=Velocity(c)
    count=lambda m:sum(p.numel() for p in m.parameters())
    encoder=count(ae.encoder);decoder=count(ae.decoder);observation=count(v.observation_encoder);context=count(v.context_encoder)
    graph=count(ae.graph_encoder);evolution=count(ae.evolution);coefficient=count(ae.coefficient)
    return {'encoder':encoder,'decoder':decoder,'coefficient':coefficient,'graph':graph,'evolution':evolution,'observation':observation,'context':context,'velocity':count(v)-observation-context,
            'total_training':count(ae)+count(v),'inference':count(ae)-encoder+count(v),'representation':encoder+decoder+coefficient,'flow':count(v),
            'trainable_A1':encoder+decoder+coefficient,'trainable_A2':graph+evolution,'trainable_B':count(v),
            'schema':schema(c),'method':'Exact module numel on meta device; one graph/evolution shared by both families. No GPU-fit claim.'}
