"""Per-boundary host clocks and CUDA stream intervals, never cumulative step costs."""
from contextlib import contextmanager
from collections import defaultdict
import time
import torch
from pdeflow.runtime import synchronize


class Timings:
    def __init__(self,device,enabled=False):
        self.device=device;self.enabled=enabled;self.cumulative=defaultdict(float);self.reset()
    def reset(self):
        self.values=defaultdict(float);self.exclusive=defaultdict(float);self.events=[];self.stack=[];self.reported=False;self.gpu=defaultdict(float)
    @contextmanager
    def measure(self,name,host=False):
        if not self.enabled:yield;return
        start=time.perf_counter();entry=[name,0.];self.stack.append(entry)
        events=self.device.type=='cuda' and not host
        if events:
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
        try:yield
        finally:
            if events:end.record();self.events.append((name,begin,end))
            duration=time.perf_counter()-start;self.stack.pop();self.values[name]+=duration;self.exclusive[name]+=duration-entry[1]
            if self.stack:self.stack[-1][1]+=duration
    def report(self):
        if not self.enabled:return None
        # Exactly one boundary synchronization, no per-reference/per-kernel sync.
        synchronize(self.device)
        for name,a,b in self.events:self.gpu[name]+=a.elapsed_time(b)/1000
        self.events=[]
        if not self.reported:
            for k,v in self.values.items():self.cumulative[k]+=v
        self.reported=True
        return {'host_inclusive_seconds':dict(self.values),'host_exclusive_seconds':dict(self.exclusive),
                'cuda_stream_interval_seconds':dict(self.gpu),'cumulative_host_inclusive_seconds':dict(self.cumulative),
                'method':'Per-step deltas. Inclusive nested scopes must NOT be added. Exclusive HOST scopes partition instrumented host time; uninstrumented time and boundary GPU wait are residual. CUDA events measure stream intervals, not host costs or a sum of kernel durations. Synchronization only at update/profiling boundaries.'}


def distribution(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    return {'count':len(a),'median':float(np.median(a)),'mean':float(a.mean()),'std':float(a.std()),'min':float(a.min()),'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9)),'max':float(a.max())}


def host_peak_bytes():
    """Process lifetime RSS high water mark; unavailable on some platforms."""
    try:
        import resource,sys
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
    except ImportError:return None
