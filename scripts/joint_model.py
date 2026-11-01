"""Configure/count the shared model or run a bounded joint-inference demonstration."""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
from pdeflow.runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['config','counts','demo']);p.add_argument('--preset',default='smoke',choices=['smoke','xsmall','small','medium','large','xlarge','custom']);p.add_argument('--device',default='cpu',choices=['cpu','mps','cuda','auto']);p.add_argument('--precision',default='fp32',choices=['fp32','bf16','fp16','auto']);p.add_argument('--output',required=True);p.add_argument('--checkpoint');p.add_argument('--family',default='joint',choices=['heat','wave','joint']);p.add_argument('--steps',type=int,default=2)
    p.add_argument('--field-loss',default='mse',choices=['mse','region_balanced_mse']);p.add_argument('--field-alpha',type=float,default=None)
    # Parse after all controls have been registered.
    args=p.parse_args()
    if args.field_alpha is not None and args.field_loss!='region_balanced_mse':p.error('--field-alpha requires region_balanced_mse')
    from pdeflow.colab import training_preset
    from pdeflow.coordinate.presets import counts,LADDER
    if args.action=='config':
        c=training_preset(args.preset,args.device,args.precision,{'field_loss':{'type':args.field_loss,**({'alpha':args.field_alpha} if args.field_alpha is not None else {})}});c=replace(c,family_weights={'heat':.5,'wave':.5} if args.family=='joint' else {args.family:1.}).validate();write_json(args.output,c.to_dict());print(__import__('pdeflow.field_losses',fromlist=['training_summary']).training_summary(c)['description']);return
    if args.action=='counts':
        write_json(args.output,{n:counts(training_preset(n,'cpu','fp32')) for n in LADDER});return
    if not args.checkpoint:p.error('demo requires --checkpoint')
    from pdeflow.coordinate.inference import reconstruct,CoordinateBundle
    from pdeflow.coordinate.dataset import example
    from pdeflow.core import GridSpec,DomainSpec
    from pdeflow.coordinate.physics import problem as dynamic_problem,get
    b=CoordinateBundle(args.checkpoint,args.device);c=b.config;root=Path(args.output);root.mkdir(parents=True,exist_ok=False);results={}
    for index,family in enumerate(sorted(c.family_weights)):
        item=example(c,'validation',index);q=np.array([[.13,.21,0.],[.37,.42,.037],[.61,.73,.237],[.82,.11,.5]]);parameter=get(family).parameter;rows=[]
        for name in get(family).state:
            values=item['solver'].evaluate(q,item['parameter'],c.interval,name)
            rows.extend(dict(field=name,x=float(x),y=float(y),time=float(t),value=float(v),sigma=0.) for (x,y,t),v in zip(q,values))
        write_json(root/(family+'-observations.json'),rows);grid=GridSpec((16,16),DomainSpec(c.lengths,c.origins),(0.,.037,.237,.5,1.))
        for known in (True,False):
            key=family+('-known' if known else '-unknown');p=dynamic_problem(grid,family,item['parameter'] if known else None);write_json(root/(key+'-problem.json'),p.to_dict());report=reconstruct(args.checkpoint,p,rows,root/key,steps=args.steps,particles=1,device=args.device)
            with np.load(root/key/'result.npz') as data:truth=item['solver'].evaluate(data['queries'],item['parameter'],c.interval);mse=float(np.mean((data['samples']-truth)**2))
            results[key]={'inference':report,'evaluation_only':{'parameter_truth':item['parameter'],'whole_field_mse':mse,'meaning':'Held-out smoke result; no useful accuracy or identification claimed'}}
    write_json(root/'joint-demo.json',results)
if __name__=='__main__':main()
