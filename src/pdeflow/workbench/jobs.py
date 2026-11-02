"""Single-compute-process queue with durable records and terminal cleanup."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import re
import threading
import time
import traceback
import uuid

from pdeflow.runtime import read_json, write_json

ACTIVE = {"queued", "running"}


def contained(root, identifier):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Malformed opaque ID")
    root = Path(root).resolve()
    path = (root / identifier).resolve()
    if path.parent != root:
        raise ValueError("Path outside configured root")
    return path


def compute_job(directory, checkpoint):
    from pdeflow.local import run_lock
    # Storage deletion and training use the same kernel ownership lock.
    with run_lock(Path(checkpoint).parent):
        return _compute_job(directory,checkpoint)


def _compute_job(directory, checkpoint):
    """Child has no reference argument or access through its request schema."""
    from pdeflow.core import ProblemSpec
    from pdeflow.observations import ObservationSet
    from pdeflow.runtime import RuntimeSpec, InferenceConfig, environment, file_digest
    from pdeflow.training import load_inference
    from pdeflow.sampling import sample
    from pdeflow.evaluation import export_result
    directory = Path(directory)
    request = read_json(directory / "request.json")
    if request.get("kind")=="mesh":
        from pdeflow.mesh.inference import compute_job as mesh_job
        return mesh_job(directory,checkpoint)
    if request.get("kind")=="coordinate":
        from pdeflow.coordinate.inference import compute_job as coordinate_job
        return coordinate_job(directory,checkpoint)
    (directory / "worker.log").write_text("Loading server-owned checkpoint; reference-free reconstruction\n")
    record_path = directory / "job.json"

    def update(**changes):
        record = read_json(record_path)
        record.update(changes)
        write_json(record_path, record)

    try:
        update(status="running", stage="loading_checkpoint", progress=0.)
        runtime = RuntimeSpec(**request["runtime"])
        if file_digest(checkpoint)[:32] != request["model_id"]:
            raise ValueError("Checkpoint changed after selection; rediscover and validate the current export")
        bundle = load_inference(checkpoint, runtime=runtime)
        if bundle.metadata["checkpoint_sha256"][:32] != request["model_id"]:
            raise ValueError("Checkpoint changed while loading; rediscover and validate the current export")
        problem = ProblemSpec.from_dict(request["problem"])
        observations = ObservationSet.from_dict(request["observations"], problem.grid)
        inference = InferenceConfig(**request["inference"])
        update(stage="sampling")

        def progress(step, total):
            update(stage="sampling", completed_steps=step, total_steps=total, progress=.9 * step / total)

        fine=None
        if request.get('refinement_checkpoint'):
            fine=load_inference(request['refinement_checkpoint'],runtime=runtime)
            if fine.metadata['checkpoint_sha256'][:32]!=request['refinement_model_id']:raise ValueError('Fine checkpoint changed after selection')
        def process_event(event):
            update(stage=event['stage'],process_event=event)
            write_json(directory/'process-live.json',event)
        result = sample(bundle, problem, observations, inference, request["mode"], progress=progress,fine_bundle=fine,process_event=process_event)
        if 'final_problem' in result.provenance:
            problem=ProblemSpec.from_dict(result.provenance['final_problem'])
            observations=ObservationSet.from_dict(result.provenance['final_observations'],problem.grid)
        update(stage="exporting", progress=.95)
        export_result(directory / "result", result, problem, observations)  # deliberately no reference
        from pdeflow.specialists import get_specialist
        update(stage='conventional_baseline')
        for name,baseline in get_specialist(problem.family).baseline(problem,observations,inference).items():
            export_result(directory/('baseline_'+name),baseline,problem,observations)
        with (directory / "worker.log").open("a") as log:
            log.write(f"Completed {inference.steps} generation steps; {result.diagnostics['model_evaluations']} batched model calls\n")
        write_json(directory / "environment.json", environment(runtime))
        update(status="failed" if result.stop_reason=="FAIL_INVALID_NUMERICS" else "succeeded", stage="completed", progress=1., finished_at=time.time(), stop_reason=result.stop_reason)
    except BaseException as error:
        (directory / "worker.log").write_text(traceback.format_exc())
        update(status="failed", stage="failed", error={"code": type(error).__name__, "message": str(error)}, finished_at=time.time())


class JobManager:
    def __init__(self, root, worker=compute_job):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.worker = worker
        self.lock = threading.RLock()
        self.stopping = threading.Event()
        self.process = None
        self.active_id = None
        self.checkpoints = {}
        for p in self.root.glob("*/job.json"):
            record = read_json(p)
            if record["status"] in ACTIVE:
                record.update(status="interrupted", stage="server_restart", finished_at=time.time())
                write_json(p, record)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def get(self, identifier):
        return read_json(contained(self.root, identifier) / "job.json")

    def list(self):
        return sorted([read_json(p) for p in self.root.glob("*/job.json")], key=lambda x: x["created_at"], reverse=True)

    def submit(self, request, checkpoint, token):
        with self.lock:
            # Token is an idempotency key: identical retries return the existing job.
            from pdeflow.runtime import digest
            request_hash = digest(request)
            for record in self.list():
                if record.get("submission_token") == token:
                    if record["request_hash"] != request_hash:
                        raise ValueError("Submission token reused with a different request")
                    return record
            identifier = uuid.uuid4().hex
            directory = contained(self.root, identifier)
            directory.mkdir()
            write_json(directory / "request.json", request)
            record = {"id": identifier, "status": "queued", "stage": "queued", "progress": 0., "created_at": time.time(),
                      "submission_token": token, "request_hash": request_hash, "model_id": request["model_id"],
                      "seed": request["inference"]["seed"]}
            if all(key in request for key in ('mode','problem','runtime')):record['summary']={'mode':request['mode'],'model_id':request['model_id'],'grid':request['problem']['grid']['shape'],'device':request['runtime']['device']}
            write_json(directory / "job.json", record)
            self.checkpoints[identifier] = str(checkpoint)
            return record

    def _loop(self):
        while not self.stopping.wait(.05):
            with self.lock:
                if self.process is not None and not self.process.is_alive():
                    self.process.join()
                    record = self.get(self.active_id)
                    if record["status"] in ACTIVE:
                        record.update(status="failed", stage="worker_exit", error={"code": "WorkerExit", "message": f"Worker exited {self.process.exitcode}"})
                        write_json(contained(self.root, self.active_id) / "job.json", record)
                    self.process.close(); self.process = None; self.active_id = None
                if self.process is None:
                    queued = sorted((r for r in self.list() if r["status"] == "queued"), key=lambda x: x["created_at"])
                    if queued:
                        record = queued[0]; identifier = record["id"]
                        self.process = mp.get_context("spawn").Process(target=self.worker, args=(str(contained(self.root, identifier)), self.checkpoints[identifier]))
                        self.active_id = identifier
                        self.process.start()

    def cancel(self, identifier):
        with self.lock:
            record = self.get(identifier)
            if record["status"] not in ACTIVE:
                return record
            if identifier == self.active_id and self.process is not None:
                self.process.terminate(); self.process.join(timeout=5)
                if self.process.is_alive(): self.process.kill(); self.process.join()
                self.process.close(); self.process = None; self.active_id = None
            record = self.get(identifier)
            if record["status"] not in ACTIVE:
                return record
            record.update(status="cancelled", stop_reason="STOP_CANCELLED", stage="cancelled", finished_at=time.time())
            write_json(contained(self.root, identifier) / "job.json", record)
            return record

    def close(self):
        self.stopping.set(); self.thread.join(timeout=2)
        with self.lock:
            if self.process is not None:
                self.process.terminate(); self.process.join(timeout=5)
                if self.process.is_alive(): self.process.kill(); self.process.join()
                self.process.close(); self.process = None
            for record in self.list():
                if record["status"] in ACTIVE:
                    record.update(status="interrupted", stage="server_shutdown", finished_at=time.time())
                    write_json(contained(self.root, record["id"]) / "job.json", record)
