"""Compatibility entry point for shared physical-state validation."""
from .validation import representation,evolution,flow,temporal_errors,evaluation_grid

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--subset',type=int,default=2);p.add_argument('--count',type=int,default=12);a=p.parse_args();flow(a.checkpoint,a.output,a.device,subset=a.subset,counts=[a.count],particles=1,steps=2)
if __name__=='__main__':main()
