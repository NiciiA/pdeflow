"""Package portable source; no scientific artifacts, caches or personal files."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pdeflow.source_bundle import package,verify

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    result=package(args.output);verify(args.output);print(result)
