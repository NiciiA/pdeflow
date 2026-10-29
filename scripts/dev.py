"""Launch local API and Vite; terminate both child process groups on exit."""
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    from pdeflow.local import require_node
    require_node()
    root=Path(__file__).resolve().parents[1]
    processes=[]
    try:
        processes.append(subprocess.Popen([sys.executable,'-m','pdeflow.workbench.api'],cwd=root,start_new_session=True))
        processes.append(subprocess.Popen(['npm','run','dev'],cwd=root/'frontend',start_new_session=True))
        print('Workbench http://127.0.0.1:5173 · API http://127.0.0.1:8000 · Ctrl-C stops both',flush=True)
        while True:
            for p in processes:
                if p.poll() is not None:raise RuntimeError(f'Child exited {p.returncode}')
            import time
            time.sleep(.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()


if __name__=='__main__':main()
