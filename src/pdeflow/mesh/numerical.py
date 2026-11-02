"""Small explicit precision boundaries; no clipping or replacement of invalid values."""
import torch
from contextvars import ContextVar

trace_sink=ContextVar("mesh_numerical_trace",default=None)

def record_tensor(name,value):
    sink=trace_sink.get()
    if sink is not None:sink(name,value)
    return value


def fp32_matmul(left,right):
    # .float() alone is insufficient: matmul is autocast-eligible even for FP32 inputs.
    with torch.autocast(left.device.type,enabled=False):
        return record_tensor("FP32 matrix product",left.float()@right.float())


def squared_error(prediction,target):
    with torch.autocast(prediction.device.type,enabled=False):
        return (prediction.float()-target.float()).square()
