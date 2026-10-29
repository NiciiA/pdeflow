"""Bounded orchestration, portable safe checkpoints and exact same-config resume."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time
import pickle
import signal
import threading

import torch

from pdeflow.core import GridSpec, TransformBundle
from pdeflow.specialists import get_specialist
from pdeflow.data import load_dataset
from pdeflow.generative import make_formulation
from pdeflow.models import TimeChannelLayout, make_model
from pdeflow.runtime import Config, atomic_write, code_identity, digest, environment, file_digest, process_peak_memory_bytes, synchronize, write_json


@dataclass
class InferenceBundle:
    model: torch.nn.Module
    transform: TransformBundle
    layout: TimeChannelLayout
    config: Config
    metadata: dict


def _read_checkpoint(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except (EOFError, RuntimeError, pickle.UnpicklingError) as error:
        raise ValueError("Checkpoint is incomplete or not a supported safe checkpoint") from error
    if ckpt.get("schema_version") != 1 or ckpt.get("kind") not in ("resume", "inference"):
        raise ValueError("Unsupported checkpoint schema/kind")
    specialist = get_specialist(ckpt.get("config", {}).get("specialist", "heat"))
    if digest(ckpt.get("capabilities")) != digest(specialist.capabilities):
        raise ValueError("Checkpoint capability declaration mismatch")
    config = Config.from_dict(ckpt["config"], check_device=False)
    if ckpt["layout"]["channels"] != len(specialist.fields) or ckpt["layout"]["fields"] != list(specialist.fields) or GridSpec.from_dict(ckpt["layout"]["grid"]) != config.grid:
        raise ValueError("Checkpoint channel/time/grid layout mismatch")
    if ckpt["formulation"] != config.formulation or ckpt["conditioning"] != specialist.conditioning_schema:
        raise ValueError("Checkpoint formulation/conditioning mismatch")
    transform = specialist.transform_type(**{k: tuple(v) for k, v in ckpt["transform"].items()})
    if list(transform.fitted_ids) != ckpt["training_ids"] or len(transform.mean) != len(specialist.fields):
        raise ValueError("Checkpoint normalization provenance mismatch")
    return ckpt, config, transform


def load_inference(path, runtime=None, expected: Config | None = None) -> InferenceBundle:
    ckpt, config, transform = _read_checkpoint(path)
    if expected is not None:
        for k in ("grid", "model", "formulation", "specialist"):
            if digest(expected.to_dict()[k]) != digest(config.to_dict()[k]):
                raise ValueError(f"Inference checkpoint {k} mismatch")
    device, dtype = (runtime or config.runtime).resolve()
    specialist = get_specialist(config.specialist)
    layout = TimeChannelLayout(len(specialist.fields), config.grid)
    model = make_model(config.model, layout, specialist.encoder).to(device=device, dtype=dtype)
    if ckpt.get('architecture') and ckpt['architecture']!=__import__('pdeflow.models',fromlist=['architecture_metadata']).architecture_metadata(model,config.model,layout):raise ValueError('Checkpoint architecture metadata/parameter count mismatch')
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # input gradients through the model remain available
    return InferenceBundle(model, transform, layout, config, {"checkpoint_sha256": file_digest(path),
                           "step": ckpt["progress"], "capabilities": ckpt["capabilities"], "environment": ckpt["environment"],
                           "code": ckpt.get("code"), "physical_metadata": ckpt.get("physical_metadata")})


def make_optimizer(config, parameters):
    if config.optimizer != "adam":
        raise ValueError("Only Adam is implemented")
    return torch.optim.Adam(parameters, lr=config.learning_rate)


def train(config: Config, data_path, output, resume=None, stop_after=None, evaluation_path=None,*,event=None,resume_changes=False,cache_calibration=False,track_best=False,should_stop=None):
    total_start = time.perf_counter()
    config.validate()
    output = Path(output)
    if resume is None and (output / "checkpoint.pt").exists():
        raise FileExistsError("Training checkpoint exists; use --resume or a fresh output directory")
    device, dtype = config.runtime.resolve()
    code = code_identity()
    saved = _read_checkpoint(resume) if resume else None
    continuation=list(saved[0].get('continuations',[])) if saved else []
    if saved:
        ckpt,saved_config,_=saved
        previous=saved_config.to_dict();current=config.to_dict()
        if resume_changes:
            if current['training']['steps']<previous['training']['steps']:raise ValueError('Resume cannot shrink the recorded total update budget')
            previous['training']['steps']=current['training']['steps']
            previous['runtime']['device']=current['runtime']['device']
        if ckpt['kind']!='resume' or digest(previous)!=digest(current):raise ValueError('Resume requires the same resolved configuration; only explicit budget extension/device remapping is supported')
        if ckpt['code']['source_sha256']!=code['source_sha256']:raise ValueError('Resume implementation changed; use the source recorded in the checkpoint, or start a new run')
        if digest(saved_config.to_dict())!=digest(current):continuation.append({'from_steps':saved_config.training.steps,'to_steps':config.training.steps,'from_device':saved_config.runtime.device,'to_device':config.runtime.device,'scheduler':'none; existing Adam constant learning rate and optimizer state retained'})
    from pdeflow.data.providers import PhysicalProvider, LegacyOfflineProvider
    setup_start = time.perf_counter()
    data = PhysicalProvider(config,evaluation_path,output,data_path,saved[0].get('provider_state') if saved else None,saved[2] if saved else None,cache_calibration,event) if config.provider else LegacyOfflineProvider(config,data_path)
    training_ids = list(data.transform.fitted_ids)
    setup_seconds = time.perf_counter()-setup_start
    torch.manual_seed(config.training.seed)
    specialist = get_specialist(config.specialist)
    layout = TimeChannelLayout(len(specialist.fields), config.grid)
    model = make_model(config.model, layout, specialist.encoder).to(device=device, dtype=dtype)
    formulation = make_formulation(config.formulation)
    optimizer = make_optimizer(config.training, model.parameters())
    streams = {name: torch.Generator(device="cpu").manual_seed(config.training.seed + j + 1)
               for j, name in enumerate(("batch", "source", "generation"))}
    transform = data.transform
    progress, history, elapsed_before = 0, [], 0.
    if resume:
        ckpt, saved_config, saved_transform = saved
        if ckpt["dataset_sha256"] != data.metadata["sha256"] or ckpt["training_ids"] != training_ids or saved_transform != transform:
            raise ValueError("Resume dataset/split/transform mismatch")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        for name, generator in streams.items():
            generator.set_state(ckpt["rng"][name])
        torch.set_rng_state(ckpt["rng"]["torch_cpu"])
        if device.type == "cuda" and 'torch_cuda' in ckpt['rng']:
            torch.cuda.set_rng_state_all(ckpt["rng"]["torch_cuda"])
        if device.type == "mps" and 'torch_mps' in ckpt['rng']:
            torch.mps.set_rng_state(ckpt["rng"]["torch_mps"])
        progress, history, elapsed_before = ckpt["progress"], ckpt["history"], ckpt["training_seconds"]
    target = config.training.steps if stop_after is None else min(config.training.steps, stop_after)
    if target < progress or target < 0:
        raise ValueError("Stop budget must be >= saved progress and nonnegative")
    cost = {'provider_setup_seconds':setup_seconds,'transform_transfer_seconds':0.,'neural_host_seconds':0.,'validation_checkpoint_seconds':0.,'timing_scope':'Host wall times; accelerator work enqueued asynchronously, synchronized at checkpoint/total boundaries. Nested provider costs are not additive to data_wait.'}
    if saved:
        previous=saved[0].get('cost_accounting',{})
        for key in ['transform_transfer_seconds','neural_host_seconds','validation_checkpoint_seconds']:cost[key]=previous.get(key,0.)
        if not config.provider:data.presentations=saved[0].get('provider_state',{}).get('presentations',0)
    interrupted=False
    signal_enabled=threading.current_thread() is threading.main_thread()
    previous_handler=signal.getsignal(signal.SIGINT) if signal_enabled else None
    def defer_interrupt(signum,frame):
        nonlocal interrupted
        interrupted=True
    def batch(physical,condition):
        before=time.perf_counter()
        values = layout.pack(transform.encode(physical)).to(device=device, dtype=dtype)
        condition=condition.to(device=device, dtype=dtype)
        cost['transform_transfer_seconds']+=time.perf_counter()-before
        return values,condition
    synchronize(device)
    start = time.perf_counter()

    def save(step):
        synchronize(device)
        rng = {name: g.get_state() for name, g in streams.items()}
        rng["torch_cpu"] = torch.get_rng_state()
        if device.type == "cuda":
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        if device.type == "mps":
            rng["torch_mps"] = torch.mps.get_rng_state()
        checkpoint = {"schema_version": 1, "kind": "resume", "model": model.state_dict(), "config": config.to_dict(),
                      "continuations":continuation,
                      "formulation": formulation.name, "conditioning": specialist.conditioning_schema,
                      "architecture": __import__("pdeflow.models",fromlist=["architecture_metadata"]).architecture_metadata(model,config.model,layout),
                      "transform": asdict(transform), "capabilities": specialist.capabilities,
                      "layout": {"channels": len(specialist.fields), "fields": list(specialist.fields), "grid": asdict(config.grid)},
                      "dataset_sha256": data.metadata["sha256"], "training_ids": training_ids,
                      "split_ids": {s: [data.metadata["ids"][i] for i in ix] for s, ix in data.metadata["splits"].items()},
                      "provider_state":data.state_dict(), "cost_accounting":cost.copy(),
                      "progress": step, "optimizer": optimizer.state_dict(), "scheduler": None, "scaler": None,
                      "rng": rng, "history": history, "training_seconds": elapsed_before + time.perf_counter() - start,
                      "environment": environment(config.runtime), "code": code,
                      "physical_metadata": {"field_units": {name:"nondimensional" for name in specialist.fields}, "time_units": "not_applicable" if config.grid.temporal_kind == "static" else "nondimensional",
                                            "unknowns": list(specialist.fields), "knowns": specialist.conditioning_schema,
                                            "boundary": "periodic", "normalization": type(transform).__name__, "representation": specialist.representation}}
        io_start=time.perf_counter()
        atomic_write(output / "checkpoint.pt", lambda f: torch.save(checkpoint, f))
        export = {k: v for k, v in checkpoint.items() if k not in ("optimizer", "scheduler", "scaler", "rng", "history")}
        export["kind"] = "inference"
        atomic_write(output / "inference.pt", lambda f: torch.save(export, f))
        if track_best and history and 'validation_loss' in history[-1]:
            candidates=[row for row in history if 'validation_loss' in row]
            best=min(candidates,key=lambda row:row['validation_loss'])
            if best['step']==step:atomic_write(output/'best-inference.pt',lambda f:torch.save(export,f))
        cost["validation_checkpoint_seconds"]+=time.perf_counter()-io_start
        write_json(output / "history.json", history)
        write_json(output / "resolved_config.json", config.to_dict())
        write_json(output / "manifest.json", {"progress": step, "training_seconds": checkpoint["training_seconds"],
                   "stop_reason": "interrupt_saved_at_safe_boundary" if interrupted else "max_steps" if step == config.training.steps else "bounded_interruption",
                   "data_provider":data.report(), "cost_accounting":{**cost,"total_session_seconds":time.perf_counter()-total_start},
                   "environment": checkpoint["environment"], "dataset_sha256": data.metadata["sha256"],
                   "process_peak_rss_bytes": process_peak_memory_bytes(), "memory_scope": "process_lifetime_high_water_mark"})
        if event:event({'kind':'checkpoint','stage':'Saving checkpoint','step':step,'total':config.training.steps,'interrupted':interrupted,'latest':history[-1] if history else None,'last_validation':next((row for row in reversed(history) if 'validation_loss' in row),None),'seconds':time.perf_counter()-start,'data_wait_seconds':data.report().get('accounting',{}).get('data_wait_seconds',0.),'checkpoint':str(output/'checkpoint.pt')})

    model.train()
    if not resume:save(0)
    if signal_enabled:signal.signal(signal.SIGINT,defer_interrupt)
    completed = progress
    if event:event({'kind':'stage','stage':'Training','step':progress,'total':config.training.steps})
    try:
      for step in range(progress + 1, target + 1):
          if should_stop and should_stop():interrupted=True
          if interrupted:break
          train_values, train_condition = batch(*data.training_batch(streams["batch"],config.training.batch_size))
          neural_start=time.perf_counter()
          x, velocity_target, s = formulation.training_state(train_values, streams["source"], streams["generation"])
          prediction = model(x, s, train_condition)
          loss = formulation.loss(prediction, velocity_target)
          if not torch.isfinite(loss):
              raise FloatingPointError(f"Nonfinite training loss at step {step}")
          optimizer.zero_grad(set_to_none=True)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
          optimizer.step()
          completed=step
          cost["neural_host_seconds"]+=time.perf_counter()-neural_start
          row = {"step": step, "training_loss": loss.item(), "physical_ids":list(data.last_ids)}
          validation_start=time.perf_counter()
          if step % config.training.checkpoint_every == 0 or step == target:
              model.eval()
              # Fixed validation randomness, independent of train streams and hidden test data.
              vg = torch.Generator().manual_seed(config.training.seed + 1001)
              vt = torch.Generator().manual_seed(config.training.seed + 1002)
              with torch.no_grad():
                  total = 0.
                  count=0
                  for physical,condition in data.validation_batches(config.training.batch_size):
                      val_values, val_condition = batch(physical,condition)
                      count+=len(physical)
                      vx, vy, vs = formulation.training_state(val_values, vg, vt)
                      total += formulation.loss(model(vx, vs, val_condition), vy).item() * len(physical)
                  row["validation_loss"] = total / count
              model.train()
          history.append(row)
          row['data_wait_seconds']=getattr(data,'accounting',{}).get('data_wait_seconds',0.)
          row['neural_host_seconds']=cost['neural_host_seconds']
          if event:event({'kind':'progress','step':step,'total':config.training.steps,'latest':row,'unique_physical_problems':len(set(getattr(data,'consumed_ids',[]))),'presentations':getattr(data,'presentations',None),'data_accounting':getattr(data,'accounting',{}).copy(),'elapsed_seconds':time.perf_counter()-start})
          if "validation_loss" in row:
              cost["validation_checkpoint_seconds"]+=time.perf_counter()-validation_start
              save(step)
              if event is None:print(f"step={step} train={loss.item():.6f} validation={row['validation_loss']:.6f}", flush=True)
      if completed == progress or interrupted:
          save(completed)
      write_json(output/'data-accounting.json',data.report())
    finally:
      if signal_enabled:signal.signal(signal.SIGINT,previous_handler)
      data.close()
    return {"progress": completed, "history": history, "output": str(output),"interrupted":interrupted,"cost":{**cost,"total_session_seconds":time.perf_counter()-total_start}}
