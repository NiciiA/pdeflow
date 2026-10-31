#!/usr/bin/env python3
"""Plan a controlled representation-loss comparison; execution requires --run."""
import argparse
from dataclasses import replace
from pathlib import Path
from pdeflow.colab import training_preset
from pdeflow.coordinate.training import train
from pdeflow.coordinate.validation import representation
from pdeflow.runtime import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--preset',choices=['pipeline-check','smoke','xsmall','small','medium','large','xlarge'],default='xsmall')
    p.add_argument('--updates',type=int,default=200);p.add_argument('--run',action='store_true');args=p.parse_args()
    if args.updates<1:raise ValueError('Updates must be positive')
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Choose a fresh comparison root; existing plans/results are preserved')
    base=training_preset(args.preset,args.device,'auto')
    base=replace(base,ae_steps=args.updates,checkpoint_every=min(50,args.updates),training_options={**base.training_options,'validation_every':min(50,args.updates),'autoencoder':{**base.training_options.get('autoencoder',{}),'warmup':min(100,args.updates//10),'horizon':args.updates}})
    plans={}
    for kind in ('mse','weighted_mse','region_balanced_mse','mae'):
        c=replace(base,field_loss={**base.field_loss,'type':kind,'alpha':.5})
        directory=args.output/kind;directory.mkdir(parents=True,exist_ok=True);write_json(directory/'planned-config.json',c.to_dict());plans[kind]=c.to_dict()
        if args.run:
            train(c.to_dict(),directory/'representation','autoencoder')
            representation(directory/'representation/autoencoder.pt',directory/'validation.json',args.device,subset=min(24,c.validation_count))
    write_json(args.output/'comparison-plan.json',{'configs':plans,'executed':args.run,'selection':'Validation only; no test cases or winning budget chosen',
        'controlled':'Same architecture, underlying solution/split/normalization identities, independent domain/region RNGs, physical-time groups, observations, optimizer and update budget.',
        'cost':'Region-balanced adds its recorded fixed region-query budget separately for solution and initial fields. Compare measured query counts/data_seconds/step_seconds; equal updates are not equal compute.',
        'metrics':'Uniform physical MSE/RMSE/MAE/relative L2, weighted/region errors, per-region size/error, background, initial/time strata and spectral power distortion.'})
    print('Executed comparison' if args.run else 'Plans only; no training launched',args.output)
if __name__=='__main__':main()
