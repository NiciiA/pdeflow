"""Install the package while retaining a compatible Colab PyTorch/CUDA stack."""
import argparse,importlib.metadata,json,subprocess,sys
from pathlib import Path

def verify():
    if sys.version_info[:2] not in ((3,11),(3,12)):raise RuntimeError('Use Python 3.11/3.12 (e.g. Colab runtime 2026.07). Do not replace a running interpreter in place.')
    try:import torch,numpy,matplotlib
    except ImportError as error:raise RuntimeError('Select a supported Colab runtime with PyTorch, NumPy and Matplotlib installed. Setup preserves the runtime CUDA stack and does not install/replace PyTorch.') from error
    from packaging.version import Version
    for name,version,lo,hi in [('torch',torch.__version__,'2.5','3'),('numpy',numpy.__version__,'2.0','3'),('matplotlib',matplotlib.__version__,'3.9','4')]:
        if not Version(lo)<=Version(version.split('+')[0])<Version(hi):raise RuntimeError(f'Unsupported {name} {version}; expected >= {lo}, < {hi}. Select a compatible runtime; setup does not replace PyTorch.')
    return {'python':sys.version,'torch':str(torch.__version__),'cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'numpy':numpy.__version__,'matplotlib':matplotlib.__version__}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    report=verify()
    if not args.check_only:
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','-e',str(Path(__file__).resolve().parents[1])],check=True)
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
