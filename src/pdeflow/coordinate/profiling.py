"""Opt-in component timings, synchronized events on CUDA, bounded host timings."""
from contextlib import contextmanager
import time
import torch
from pdeflow.runtime import synchronize

class Timings:
    def __init__(self,device,enabled=False):self.device=device;self.enabled=enabled;self.values={};self.events=[]
    def reset(self):self.values={};self.events=[]
    @contextmanager
    def measure(self,name,host=False):
        if not self.enabled:yield;return
        if self.device.type=='cuda' and not host:
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
            try:yield
            finally:end.record();self.events.append((name,begin,end))
        else:
            synchronize(self.device) if not host else None;start=time.perf_counter()
            try:yield
            finally:synchronize(self.device) if not host else None;self.values[name]=self.values.get(name,0.)+time.perf_counter()-start
    def report(self):
        if not self.enabled:return None
        synchronize(self.device)
        for name,a,b in self.events:self.values[name]=self.values.get(name,0.)+a.elapsed_time(b)/1000
        return {'seconds':self.values.copy(),'method':'CUDA events; CPU/MPS synchronized wall time. Nested/inclusive components must not be added; opt-in overhead. First update is warmup.'}
