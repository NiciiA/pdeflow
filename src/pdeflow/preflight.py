"""A bounded resource/gradient/persistence check, never a quality benchmark."""
from dataclasses import asdict, replace
from pathlib import Path
import time
import torch

from pdeflow.core import ProblemSpec
from pdeflow.data import generate, load_dataset
from pdeflow.observations import ObservationSet
from pdeflow.runtime import RuntimeSpec, environment, process_peak_memory_bytes, read_json, synchronize, write_json
from pdeflow.sampling import sample
from pdeflow.training import load_inference, train


def preflight(config, output, data_path=None):
    config.validate();output=Path(output)
    if output.exists():raise FileExistsError('Preflight output exists; choose a fresh directory')
    output.mkdir(parents=True);start=time.perf_counter();device,_=config.runtime.resolve()
    memory_before=torch.mps.current_allocated_memory() if device.type=='mps' else None
    if config.provider:
        from pdeflow.data.providers import prepare
        evaluation=output/'evaluation';prepare(config,evaluation)
        data_path=output/'data' if config.provider.mode=='offline' else None
        if data_path:prepare(config,data_path,True)
        train(config,data_path,output/'training',stop_after=1,evaluation_path=evaluation)
        train(config,data_path,output/'training',resume=output/'training/checkpoint.pt',stop_after=2,evaluation_path=evaluation)
        inputs=evaluation/'test/0';problem=ProblemSpec.from_dict(read_json(inputs/'problem.json'))
        observations=ObservationSet.from_dict(read_json(inputs/'observations.json'),problem.grid)
        inf=replace(config.inference,steps=2,particles=1)
        result=sample(load_inference(output/'training/inference.pt',config.runtime),problem,observations,inf,'observation_pde')
        cpu=sample(load_inference(output/'training/inference.pt',RuntimeSpec(device='cpu')),problem,observations,inf,'observation_pde')
        if not torch.isfinite(result.samples).all() or not torch.isfinite(cpu.samples).all():raise FloatingPointError('Preflight nonfinite fields')
        report={'backend':environment(config.runtime),'actual_optimizer_updates':2,'mode':config.provider.mode,'evaluation_identity':read_json(evaluation/'manifest.json')['identity'],'provider':read_json(output/'training/data-accounting.json'),'guidance_gradients':'passed','cpu_reload':'passed','resume':'passed','quality_claim':False,'seconds':time.perf_counter()-start}
        write_json(output/'preflight.json',report);return report
    if data_path is None:
        # Explicitly smaller instance set, same grid/time/model/batch/distribution.
        config=replace(config,data=replace(config.data,count=12,split_counts=(8,2,2),shard_size=4,storage='sharded_npz'))
        data_path=output/'data';generate(config,data_path)
    data_path=Path(data_path);data=load_dataset(data_path,config)
    write_json(output/'preflight_config.json',config.to_dict())
    train(config,data_path,output/'training',stop_after=1)
    train(config,data_path,output/'training',resume=output/'training/checkpoint.pt',stop_after=2)
    problem=ProblemSpec.from_dict(read_json(data_path/'test_problem.json'))
    observations=ObservationSet.from_dict(read_json(data_path/'test_observations.json'),problem.grid)
    inference=replace(config.inference,steps=2,particles=1)
    bundle=load_inference(output/'training/inference.pt',config.runtime)
    result=sample(bundle,problem,observations,inference,'observation_pde')
    if not torch.isfinite(result.samples).all():raise FloatingPointError('Nonfinite guided samples')
    synchronize(device)
    allocated=torch.mps.current_allocated_memory() if device.type=='mps' else None
    driver=torch.mps.driver_allocated_memory() if device.type=='mps' else None
    cpu=load_inference(output/'training/inference.pt',RuntimeSpec(device='cpu'))
    cpu_result=sample(cpu,problem,observations,inference,'observation_pde')
    if not torch.isfinite(cpu_result.samples).all():raise FloatingPointError('Nonfinite CPU reload samples')
    report={'backend':environment(config.runtime),'grid':asdict(config.grid),'model':asdict(config.model),'batch_size':config.training.batch_size,
            'actual_optimizer_updates':2,'validation':'passed','resume':'passed','guidance_gradients':'passed','cpu_reload':'passed',
            'data_counts':{s:len(data.indices(s)) for s in ['train','validation','test']},'seconds':time.perf_counter()-start,
            'process_peak_rss_bytes':process_peak_memory_bytes(),'mps_allocated_before_bytes':memory_before,'mps_allocated_after_guidance_bytes':allocated,
            'mps_driver_allocated_after_guidance_bytes':driver,'memory_note':'MPS allocations are point-in-time, not peak memory; RSS is process lifetime high-water mark',
            'quality_claim':False,'source':'existing dataset' if Path(data_path).resolve()!=(output/'data').resolve() else 'explicit 12-instance preflight subset'}
    write_json(output/'preflight.json',report);return report
