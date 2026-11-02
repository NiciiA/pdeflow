"""Read-only fixed-noise inference for source-version comparisons."""
from pathlib import Path
import argparse,json
import numpy as np
import pdeflow
from pdeflow.core import GridSpec
from pdeflow.coordinate.diagnostic_cases import load_case
from pdeflow.coordinate.inference import CoordinateBundle
from pdeflow.coordinate.contracts import grid_queries
from pdeflow.coordinate.optimization import scientific_source
from pdeflow.runtime import atomic_write,write_json
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--case',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
output=Path(a.output)
if output.exists():raise FileExistsError('Existing replay is preserved; choose a fresh file')
case,problem,_,_,noise=load_case(a.case);bundle=CoordinateBundle(a.checkpoint,a.device);queries=grid_queries(GridSpec((8,8),problem.grid.domain,(0.,.237,1.)))
values,report=bundle.sample(problem,case['observations'],queries,initial_noise=noise,**{**case['inference'],'steps':4})
atomic_write(output,lambda stream:np.savez_compressed(stream,samples=values,latent=bundle.decoded_latent.numpy(),parameters=bundle.parameter.numpy(),queries=queries.numpy()))
write_json(output.with_suffix('.json'),{'source_module':str(Path(pdeflow.__file__).resolve()),'scientific_source':scientific_source()['sha256'],'checkpoint':str(Path(a.checkpoint).resolve()),'case_identity':case['case_identity'],'device':a.device,'report':report})
print(output)
