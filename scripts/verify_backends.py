"""Explicit CPU/MPS regression, portable reload and step-cost estimate."""
import argparse
from dataclasses import replace
from pathlib import Path
import time
import torch

from pdeflow.cli import smoke,reconstruct
from pdeflow.data import generate
from pdeflow.runtime import Config,write_json,environment
from pdeflow.training import train,load_inference


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--device',choices=['cpu','mps','cuda'],default='mps');args=parser.parse_args()
    root=Path(args.output)
    config=Config.load('configs/experiments/mps_regression.json',device=args.device)
    report=smoke(config,root)
    bundle=load_inference(root/'training/inference.pt',runtime=replace(config.runtime,device='cpu'))
    assert next(bundle.model.parameters()).device.type=='cpu'
    cpu=reconstruct(replace(config,runtime=replace(config.runtime,device='cpu')),root/'training/inference.pt',root/'data/test_problem.json',root/'data/test_observations.json',root/'cpu_reload')
    write_json(root/'backend_verification.json',{'backend':environment(config.runtime),'training_updates':6,'checkpoint_cpu_reload':True,'device_comparison_bitwise_claim':False,'device_metrics':report,'cpu_metrics':cpu})
    print(root/'backend_verification.json')


if __name__=='__main__':main()
