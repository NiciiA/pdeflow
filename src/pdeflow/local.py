"""Filesystem-based local orchestration over the public scientific package.

No training, reference solving or reconstruction algorithms live here.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import fcntl
import importlib.util
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import uuid

from pdeflow.runtime import Config, RuntimeSpec, code_identity, read_json, write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def project_root():
    source = Path(__file__).resolve().parents[2]
    if (source / 'frontend/package.json').is_file():
        return source
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / 'frontend/package.json').is_file() and (path / 'pyproject.toml').is_file():
            return path
    return Path.cwd()


def workspace_root(workspace=None):
    return Path(workspace or os.environ.get('PDEFLOW_WORKSPACE') or project_root() / 'outputs/local').expanduser().resolve()


def initialize(workspace=None):
    root = workspace_root(workspace)
    for directory in ['runs', 'cache', 'workbench/jobs', 'workbench/data']:
        (root / directory).mkdir(parents=True, exist_ok=True)
    return root


def preset_config(name):
    path = files('pdeflow').joinpath('presets', name + '.json')
    if name not in [r['name'] for r in presets()]:
        raise ValueError(f'Unknown preset {name!r}; run pdeflow presets')
    return Config.from_dict(json.loads(path.read_text()), check_device=False)


def presets():
    result = []
    for path in sorted(files('pdeflow').joinpath('presets').iterdir(), key=lambda p: p.name):
        if not path.name.endswith('.json'):
            continue
        c = Config.from_dict(json.loads(path.read_text()), check_device=False)
        name = path.name[:-5]
        result.append({'name': name, 'specialist': c.specialist, 'grid': list(c.grid.shape),
                       'times': list(c.grid.times), 'temporal_kind': c.grid.temporal_kind,
                       'mode': c.provider.mode, 'model': {'width': c.model.width, 'blocks': c.model.blocks},
                       'batch_size': c.training.batch_size, 'steps': c.training.steps,
                       'checkpoint_every': c.training.checkpoint_every,
                       'profile': 'quick check' if name.startswith('quickstart') else 'training pilot; accuracy unestablished',
                       'intended_device': 'CUDA / Colab' if name.startswith('colab') else 'MacBook MPS; CPU explicit; auto reports choice',
                       'hardware_evidence': 'See docs/everyday-verification.md; pilot default budgets not executed, quickstart default CPU journey exercised'})
    return result


def node_info():
    info = {'required': '^22.13.0 || ^24.0.0 || >=26.0.0', 'node': None, 'npm': None, 'compatible': False}
    for name in ['node', 'npm']:
        executable = shutil.which(name)
        if executable:
            try:
                info[name] = subprocess.run([executable, '--version'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            except (subprocess.SubprocessError, OSError):
                pass
    if info['node']:
        version = tuple(int(v) for v in info['node'].lstrip('v').split('.')[:3])
        info['compatible'] = (version[0] == 22 and version >= (22, 13, 0)) or version[0] == 24 or version[0] >= 26
    info['action'] = 'Install Node 22.23.3 (nvm install using .nvmrc), then pdeflow setup; headless training does not require Node'
    return info


def require_node():
    info = node_info()
    if not info['compatible'] or not info['npm']:
        raise ValueError(f"Frontend needs Node {info['required']} and npm; detected node={info['node']}, npm={info['npm']}. {info['action']}")
    return info


def setup(workspace=None, frontend=True):
    root = initialize(workspace)
    result = {'workspace': str(root), 'python': sys.executable, 'training_launched': False}
    if frontend:
        result['node'] = require_node()
        folder = project_root() / 'frontend'
        if not (folder / 'package-lock.json').is_file():
            raise ValueError('Frontend source/lockfile unavailable; run from a source checkout or use pdeflow setup --headless')
        # npm ci is intentionally lockfile-based. Re-running never touches runs/settings.
        try:subprocess.run(['npm', 'ci', '--no-audit', '--no-fund'], cwd=folder, check=True)
        except subprocess.CalledProcessError as error:
            raise ValueError(f'npm ci failed with exit {error.returncode}; inspect its output, verify Node with pdeflow doctor, then rerun pdeflow setup. Existing runs were preserved') from error
        result['frontend'] = 'npm ci completed using package-lock.json'
    else:
        result['frontend'] = 'skipped; headless operation ready'
    return result


def run_records(workspace=None):
    result = []
    root=workspace_root(workspace)
    paths=list((root/'runs').glob('*/run.json'))
    if (root/'run-links.json').exists():paths.extend(Path(p)/'run.json' for p in read_json(root/'run-links.json')['paths'])
    for path in sorted(set(paths)):
        try:
            record = read_json(path)
            if record.get('kind') == 'pdeflow_local_run' and record.get('schema_version') == 1:
                result.append({**record, 'path': str(path.parent)})
        except (ValueError, OSError):
            continue  # doctor reports corrupt manifests separately
    return sorted(result, key=lambda r: (r['started_at'], r['id']), reverse=True)


def resolve_run(selector='latest', workspace=None, specialist=None):
    root = workspace_root(workspace)
    if selector == 'latest':
        candidates = [r for r in run_records(root) if specialist is None or r['specialist'] == specialist]
        if not candidates:
            raise ValueError('No matching training run. Run pdeflow presets, then pdeflow train --preset quickstart-heat-2d --device cpu')
        path = Path(candidates[0]['path'])
    else:
        candidate = Path(selector).expanduser()
        path = candidate.resolve() if candidate.exists() else root / 'runs' / selector
    if not (path / 'run.json').is_file():
        raise ValueError(f'No managed run at {path}; use pdeflow runs --workspace {root}')
    record = read_json(path / 'run.json')
    if record.get('kind') != 'pdeflow_local_run' or record.get('schema_version') != 1:
        raise ValueError('Unsupported run manifest; existing advanced outputs can be linked with pdeflow models --import PATH')
    if specialist and record['specialist'] != specialist:
        raise ValueError(f"Run is {record['specialist']}, requested {specialist}; choose an explicit compatible run")
    return path, record


@contextmanager
def run_lock(path):
    """Kernel-owned advisory lock; stale metadata cannot retain a dead lock."""
    path = Path(path)
    with (path / '.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.seek(0)
            raise ValueError(f'Another process owns run {path}: {handle.read().strip()}. Wait for it or stop that process; do not delete its lock') from error
        handle.seek(0); handle.truncate()
        json.dump({'pid': os.getpid(), 'host': socket.gethostname(), 'acquired_at': now()}, handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    # Keep metadata for diagnosis; OS automatically releases locks after crashes.


def is_locked(path):
    file = Path(path) / '.run.lock'
    if not file.exists():
        return False
    with file.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
    return False


def status(selector='latest', workspace=None, specialist=None):
    path, record = resolve_run(selector, workspace, specialist)
    locked = is_locked(path)
    state = record['state']
    if state in ['preparing', 'training'] and not locked:
        state = 'interrupted (owner exited; lock released)'
    live=read_json(path/'progress.json') if (path/'progress.json').exists() else None
    heartbeat=read_json(path/'heartbeat.json') if (path/'heartbeat.json').exists() else None
    return {**record, 'live_progress':live,'current_update':max(live.get('step',0),record['progress']) if live and locked else record['progress'],'heartbeat':heartbeat,'heartbeat_fresh':(time.time()-heartbeat['unix']<15) if heartbeat else None,'state': state, 'path': str(path), 'locked': locked,
            'resume_command': f'pdeflow resume {path}'}


def _provenance():
    result = code_identity()
    try:
        p = subprocess.run(['git', '-C', str(project_root()), 'status', '--porcelain'], capture_output=True, text=True, timeout=10)
        result['git_dirty'] = bool(p.stdout) if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        result['git_dirty'] = None
    return result


def _announce(path, record):
    print(f"Run {record['id']} · {record['specialist']} · {path}", flush=True)


def _execute(path, record, config, stop_after=None, should_stop=None):
    from pdeflow.data.providers import prepare
    from pdeflow.training import train, _read_checkpoint
    from pdeflow.data import generate
    started = time.perf_counter()
    initial_step = record.get('progress', 0)
    import threading
    heartbeat_stop=threading.Event()
    def heartbeat():
        while not heartbeat_stop.is_set():
            write_json(path/'heartbeat.json',{'pid':os.getpid(),'host':socket.gethostname(),'at':now(),'unix':time.time()})
            heartbeat_stop.wait(2)
    heartbeat_thread=threading.Thread(target=heartbeat,daemon=True);heartbeat_thread.start()
    last_live=[0.];first_accounting=[None]

    with (path / 'train.log').open('a', buffering=1) as log:
        def emit(message):
            print(message, flush=True); log.write(message + '\n')
        def update(**changes):
            record.update(changes, updated_at=now())
            write_json(path / 'run.json', record)
        def event(item):
            if item['kind']=='progress':
                if time.monotonic()-last_live[0]<1 and item['step']!=item['total']:return
                last_live[0]=time.monotonic()
                elapsed=item['elapsed_seconds'];done=item['step']-initial_step
                accounting=item.get('data_accounting',{})
                if first_accounting[0] is None:first_accounting[0]=(accounting.get('accepted_reference_calls',0),elapsed)
                count0,elapsed0=first_accounting[0]
                generation_rate=(accounting.get('accepted_reference_calls',0)-count0)/(elapsed-elapsed0) if elapsed>elapsed0 else None
                write_json(path/'progress.json',{**item,'at':now(),'generated_reference_problems_per_second':generation_rate,'updates_per_second':done/elapsed if elapsed>0 else None,'eta_seconds':(item['total']-item['step'])*elapsed/done if done>=20 and elapsed>=10 else None,'timing_scope':'host wall time; validation/data wait included; accelerator kernels may be asynchronous'})
                return
            with (path / 'events.jsonl').open('a') as stream:
                stream.write(json.dumps({**item, 'at': now()}) + '\n')
            if item['kind'] == 'checkpoint':
                row = item.get('latest') or {}
                validation=item.get('last_validation') or {}
                elapsed = time.perf_counter() - started
                progress = item['step']
                update(progress=progress, latest_metrics=row, last_validation=validation, checkpoint_ready=True)
                rate = max(0, progress - initial_step) / max(elapsed, 1e-9)
                emit(f"Saving checkpoint · {progress}/{config.training.steps} · loss={row.get('training_loss', '—')} · validation={validation.get('validation_loss', '—')} at {validation.get('step', '—')} · {rate:.2f} updates/s (includes preparation) · data wait={item['data_wait_seconds']:.3f}s · {item['checkpoint']}")
            else:
                if item['stage'] == 'Training':
                    update(state='training')
                emit(item['stage'])
        try:
            update(state='preparing', device=config.runtime.device, dtype=config.runtime.dtype, budget=config.training.steps)
            emit(f"Checking environment · {config.runtime.device}/{config.runtime.dtype} · preset={record['preset']}")
            evaluation = path / 'evaluation' if config.provider else None
            data = path / 'data' if not config.provider or config.provider.mode == 'offline' else None
            if config.provider:
                emit('Preparing fixed evaluation problems (reuse immutable completed records)')
                prepare(config, evaluation, resume=True)
                if data:
                    emit('Preparing offline training problems')
                    prepare(config, data, training=True, resume=True)
            elif not (data / 'manifest.json').exists():
                emit('Preparing explicit legacy offline data')
                generate(config, data)
            checkpoint = path / 'training/checkpoint.pt'
            result = train(config, data, path / 'training', resume=checkpoint if checkpoint.exists() else None,
                           stop_after=stop_after, evaluation_path=evaluation, event=event,
                           resume_changes=True, cache_calibration=True, track_best=True, should_stop=should_stop)
            checkpoint_record, _, _ = _read_checkpoint(checkpoint)
            state = 'completed' if result['progress'] >= config.training.steps else 'interrupted' if result['interrupted'] else 'paused'
            write_json(path / 'config.json', config.to_dict())
            update(state=state, progress=result['progress'], completed_at=now() if state == 'completed' else None,
                   resolved_config=config.to_dict(), continuations=checkpoint_record.get('continuations', []), error=None)
            emit(f"{state} · saved update {result['progress']} · latest inference export ready")
            emit(f"Next: pdeflow resume {path}\n      pdeflow evaluate --run {path}\n      pdeflow serve")
            return status(str(path))
        except BaseException as error:
            # Saving a continuation's config only after success preserves the old
            # checkpoint as authoritative if preparation/restore fails.
            update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error), recovery=f'pdeflow resume {path}')
            emit(f"Stopped: {type(error).__name__}: {error}. Recover with pdeflow resume {path}")
            raise
        finally:
            heartbeat_stop.set();heartbeat_thread.join(timeout=3)


def start_training(preset=None, config_path=None, workspace=None, output=None, device=None,
                   steps=None, batch_size=None, seed=None, name=None, data_mode=None, dry_run=False, stop_after=None, should_stop=None):
    if bool(preset) == bool(config_path):
        raise ValueError('Choose exactly one --preset or --config')
    c = preset_config(preset) if preset else Config.load(config_path, check_device=False)
    changes = {key: value for key, value in {'steps': steps, 'batch_size': batch_size, 'seed': seed}.items() if value is not None}
    c = replace(c, training=replace(c.training, **changes), runtime=replace(c.runtime, device=device or c.runtime.device))
    if seed is not None:
        c = replace(c, data=replace(c.data, seed=seed))
    if data_mode:
        if c.provider is None:
            raise ValueError('--data-mode requires a physical provider profile; use an online preset')
        c = replace(c, provider=replace(c.provider, mode=data_mode))
    dev, _ = c.runtime.resolve()
    c = replace(c, runtime=replace(c.runtime, device=str(dev))).validate()
    if stop_after is not None and (stop_after < 0 or stop_after > c.training.steps):
        raise ValueError('--stop-after must be between zero and the total update budget')
    if dry_run:
        return {'preset': preset, 'workspace': str(workspace_root(workspace)), 'resolved_config': c.to_dict(),
                'stages': ['Checking environment', 'Preparing fixed evaluation problems', 'Calibrating transforms', 'Training', 'Saving checkpoint'],
                'duration_estimate': 'not measured; no training launched'}
    root = initialize(workspace)
    slug = re.sub('[^a-zA-Z0-9_-]', '-', name or preset or c.specialist)[:48]
    identifier = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + slug + '-' + uuid.uuid4().hex[:8]
    path = Path(output).expanduser().resolve() if output else root / 'runs' / identifier
    path.mkdir(parents=True, exist_ok=False)
    record = {'schema_version': 1, 'kind': 'pdeflow_local_run', 'id': identifier, 'name': name,
              'specialist': c.specialist, 'preset': preset or 'explicit-config', 'profile': 'quick check' if preset and preset.startswith('quickstart') else 'pilot; scientifically unevaluated',
              'resolved_config': c.to_dict(), 'data_mode': c.provider.mode if c.provider else 'legacy_offline',
              'device': c.runtime.device, 'dtype': c.runtime.dtype, 'source': _provenance(),
              'started_at': now(), 'updated_at': now(), 'completed_at': None, 'state': 'preparing', 'progress': 0,
              'budget': c.training.steps, 'checkpoint_ready': False,
              'paths': {'checkpoint': 'training/checkpoint.pt', 'latest_export': 'training/inference.pt', 'best_export': 'training/best-inference.pt',
                        'evaluation': 'evaluation' if c.provider else 'data', 'config': 'config.json', 'history': 'training/history.json', 'log': 'train.log', 'events': 'events.jsonl'}}
    write_json(path / 'config.json', c.to_dict()); write_json(path / 'run.json', record)
    if path.parent!=root/'runs':
        with run_lock(root):
            links=root/'run-links.json'
            old=read_json(links)['paths'] if links.exists() else []
            write_json(links,{'schema_version':1,'paths':sorted(set(old+[str(path)]))})
    _announce(path, record)
    with run_lock(path):
        return _execute(path, record, c, stop_after, should_stop)


def resume_training(selector='latest', workspace=None, specialist=None, device=None, additional_steps=None, stop_after=None, should_stop=None):
    from pdeflow.training import _read_checkpoint
    path, record = resolve_run(selector, workspace, specialist)
    _announce(path, record)
    with run_lock(path):
        checkpoint = path / record['paths']['checkpoint']
        c = _read_checkpoint(checkpoint)[1] if checkpoint.exists() else Config.load(path / 'config.json', check_device=False)
        if additional_steps is not None:
            if additional_steps < 1:
                raise ValueError('--additional-steps must be positive; it adds to the recorded TOTAL budget')
            c = replace(c, training=replace(c.training, steps=c.training.steps + additional_steps))
        c = replace(c, runtime=replace(c.runtime, device=device or c.runtime.device))
        dev, _ = c.runtime.resolve(); c = replace(c, runtime=replace(c.runtime, device=str(dev))).validate()
        return _execute(path, record, c, stop_after, should_stop)


def model_entries(workspace=None):
    """Rebuild references from manifests; validate weight files in the API/CLI."""
    root = workspace_root(workspace); result = []
    for record in run_records(root):
        if not record.get('checkpoint_ready'):
            continue
        for kind, key in [('latest', 'latest_export'), ('validation-selected', 'best_export')]:
            path = Path(record['path']) / record['paths'][key]
            if path.is_file():
                result.append({'path': str(path), 'run_id': record['id'], 'run_name': record.get('name'),
                               'export_kind': kind, 'status': record['profile'], 'run_state': status(str(record['path']))['state'], 'budget':record['budget'],'progress':record['progress']})
    if (root / 'model-links.json').exists():
        for path in read_json(root / 'model-links.json')['paths']:
            result.append({'path': path, 'export_kind': 'linked', 'status': 'external; quality not established'})
    return result


def models(workspace=None, import_path=None):
    from pdeflow.training import _read_checkpoint
    root = initialize(workspace) if import_path else workspace_root(workspace)
    if import_path:
        path = Path(import_path).expanduser().resolve()
        paths = [path] if path.is_file() else sorted(path.rglob('inference.pt'))
        if not paths:
            raise ValueError(f'No inference.pt found at {path}; link an exported checkpoint or its directory')
        for path in paths:
            checkpoint, _, _ = _read_checkpoint(path)
            if checkpoint['kind'] != 'inference':
                raise ValueError('Link an inference export, not a resumable training checkpoint')
        settings = root / 'model-links.json'
        old = read_json(settings)['paths'] if settings.exists() else []
        write_json(settings, {'schema_version': 1, 'paths': sorted(set(old + [str(p) for p in paths]))})
    rows = []
    for entry in model_entries(root):
        try:
            checkpoint, c, _ = _read_checkpoint(entry['path'])
            if checkpoint['kind'] != 'inference':
                continue
            rows.append({**entry, 'specialist': c.specialist, 'grid': list(c.grid.shape), 'update': checkpoint['progress']})
        except (ValueError, OSError, KeyError, RuntimeError):
            continue
    return rows


def evaluate_run(selector='latest', workspace=None, specialist=None, split='validation', full=False, device=None, export='latest'):
    from pdeflow.cli import reconstruct
    from pdeflow.data.providers import check_manifest
    from pdeflow.training import load_inference
    path, record = resolve_run(selector, workspace, specialist)
    _announce(path, record)
    with run_lock(path):
        c = Config.load(path / 'config.json', device=device, check_device=False)
        dev, _ = c.runtime.resolve(); c = replace(c, runtime=replace(c.runtime, device=str(dev)))
        checkpoint = path / record['paths']['best_export' if export == 'best' else 'latest_export']
        if not checkpoint.is_file():
            raise ValueError(f'No {export} export at {checkpoint}; train/resume until a checkpoint is saved')
        # Use saved immutable training inference settings; no automatic test tuning.
        bundle = load_inference(checkpoint, c.runtime, expected=c)
        evaluation = path / record['paths']['evaluation']
        if c.provider:
            if not (evaluation / 'manifest.json').exists():
                raise ValueError(f'Missing fixed evaluation manifest {evaluation}; recover preparation with pdeflow resume {path}')
            manifest = check_manifest(evaluation, c, 'fixed_evaluation')
            indices = [r['index'] for r in manifest['records'][split]]
            cases = [(str(i), evaluation / split / str(i)) for i in (indices if full else indices[:1])]
        else:
            if split != 'test':
                raise ValueError('Legacy explicit datasets expose saved test observations only; use --split test or a physical-provider preset')
            cases = [('0', evaluation)]
        destination = path / 'evaluations' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        destination.mkdir(parents=True)
        selection = {'run_id': record['id'], 'split': split, 'full': full, 'case_count': len(cases),
                     'export': export, 'checkpoint_sha256': bundle.metadata['checkpoint_sha256'],
                     'config': c.to_dict(), 'started_at': now(), 'state': 'running',
                     'purpose': 'explicit held-out test; repeated inspection must not become tuning' if split == 'test' else 'quick validation; used for inspection/model selection, not final test evidence'}
        write_json(destination / 'evaluation.json', selection)
        reports = {}
        try:
            for identifier, inputs in cases:
                prefix = '' if c.provider else 'test_'
                reports[identifier] = reconstruct(c, checkpoint, inputs / (prefix + 'problem.json'), inputs / (prefix + 'observations.json'),
                                                 destination / identifier, inputs / (prefix + 'reference.npz'))
            selection.update(state='completed', finished_at=now(), reports=reports)
        except BaseException as error:
            selection.update(state='failed', error=str(error), finished_at=now())
            raise
        finally:
            write_json(destination / 'evaluation.json', selection)
        return {'run_id': record['id'], 'output': str(destination), **selection}


def port_available(host, port):
    family = socket.AF_INET6 if ':' in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            # Match Uvicorn: a closed listener's TIME_WAIT connections must not
            # be reported as a live owner when immediately restarting.
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


def doctor(workspace=None):
    import torch
    root = workspace_root(workspace); frontend = project_root() / 'frontend'; node = node_info()
    errors = []; optional = []
    api_dependencies={name:importlib.util.find_spec(name) is not None for name in ['fastapi','uvicorn','httpx']}
    if not all(api_dependencies.values()):optional.append('Backend dependencies missing: python -m pip install -e ".[workbench]"; headless training remains available')
    if not node['compatible'] or not node['npm']:
        optional.append(node['action'])
    if not (frontend / 'node_modules/.package-lock.json').exists():
        optional.append('Frontend dependencies missing: pdeflow setup (headless training is available)')
    parent = root
    while not parent.exists():
        parent = parent.parent
    if not os.access(parent, os.W_OK):
        errors.append(f'Workspace not writable at {root}; choose --workspace PATH')
    locations={}
    for name in ['runs','cache','workbench/jobs','workbench/data']:
        target=root/name;existing=target
        while not existing.exists():existing=existing.parent
        locations[name]={'path':str(target),'writable':os.access(existing,os.W_OK)}
        if not locations[name]['writable']:errors.append(f'Output not writable at {target}; choose a writable --workspace')
    for path in (root / 'runs').glob('*/run.json'):
        try:
            read_json(path)
        except (OSError, ValueError):
            errors.append(f'Unreadable run manifest {path}; preserve and repair it explicitly')
    backends = {'cpu': True, 'mps': torch.backends.mps.is_available(), 'cuda': torch.cuda.is_available()}
    default = 'cuda' if backends['cuda'] else 'mps' if backends['mps'] else 'cpu'
    return {'ok': not errors, 'errors': errors, 'optional': optional, 'python': {'executable': sys.executable, 'version': sys.version.split()[0]},
            'package': {'importable': True, 'source': str(Path(__file__).resolve().parent)}, 'torch': torch.__version__, 'backends': backends,
            'default_device': default, 'node': node, 'api_dependencies':api_dependencies,'output_locations':locations,'frontend': {'source': frontend.is_dir(), 'dependencies': (frontend / 'node_modules/.package-lock.json').exists(), 'built': (frontend / 'dist/index.html').exists()},
            'workspace': str(root), 'writable_parent': str(parent), 'presets': [r['name'] for r in presets()], 'specialists': ['heat', 'darcy', 'chemistry'],
            'models': models(root), 'ports': {str(p): 'available' if port_available('127.0.0.1', p) else 'in use / bind unavailable' for p in [8000, 5173]}}


def serve(workspace=None, host='127.0.0.1', port=8000, reload=False):
    try:import uvicorn,fastapi
    except ImportError as error:raise ValueError('Backend dependencies unavailable; run python -m pip install -e ".[workbench]"') from error
    if not 1 <= port <= 65535:
        raise ValueError('Port must be 1..65535')
    if not port_available(host, port):
        raise ValueError(f'Port {host}:{port} already in use or unavailable. Stop its owner, or use pdeflow serve --port {port + 1}')
    root = initialize(workspace)
    os.environ['PDEFLOW_WORKSPACE'] = str(root)  # inherited by uvicorn reload child
    print(f'API http://{host}:{port} · workspace {root}\nFrontend http://127.0.0.1:5173 · in another terminal: cd frontend && npm run dev', flush=True)
    uvicorn.run('pdeflow.workbench.api:create_app', factory=True, host=host, port=port, reload=reload,
                **({'reload_dirs': [str(Path(__file__).parent)]} if reload else {}))


def display(command, report, machine=False):
    if machine:
        print(json.dumps(report, indent=2, allow_nan=False)); return
    if command == 'presets':
        for row in report:
            print(f"{row['name']:<23} {row['specialist']:<10} {str(row['grid']):<14} {len(row['times'])} physical frames · {row['mode']} · width{row['model']['width']}/{row['model']['blocks']} blocks · batch{row['batch_size']} · {row['steps']} updates · save every{row['checkpoint_every']} · {row['profile']}")
        print('Auto selects CUDA, then MPS, then CPU and reports the choice. Full pilot budgets are untested; no duration/quality promise. Use --device explicitly.')
    elif command == 'runs':
        for row in report:
            print(f"{row['id']} · {row['specialist']} · {row['state']} · {row['progress']}/{row['budget']} · {row['device']}\n  {row['path']}")
        if not report:print('No managed runs. Start: pdeflow train --preset quickstart-heat-2d --device cpu')
    elif command in ['status', 'resume']:
        print(f"{report['id']} · {report['specialist']} · {report['state']} · update {report['progress']}/{report['budget']} · {report['device']}/{report['dtype']}\n{report['path']}\n{report['resume_command']}")
        if report.get('error'):print('Last error:', report['error'])
        if report.get('live_progress'):print('Current update:',report['current_update'],'latest resumable update:',report['progress'],'heartbeat fresh:',report['heartbeat_fresh'],'live:',report['live_progress'])
        if report.get('latest_metrics'):print('Latest saved metrics:',report['latest_metrics'])
        if report.get('last_validation'):print('Last validation:',report['last_validation'])
    elif command == 'doctor':
        print(f"Python {report['python']['version']} · {report['python']['executable']}\nPackage import OK · PyTorch {report['torch']}\nBackends {report['backends']} · default {report['default_device']}\nNode {report['node']['node']} · npm {report['node']['npm']} · compatible={report['node']['compatible']}\nFrontend {report['frontend']}\nWorkspace {report['workspace']} · writable parent {report['writable_parent']}\nSpecialists {', '.join(report['specialists'])} · presets {len(report['presets'])} · checkpoints {len(report['models'])}\nPorts {report['ports']}")
        for message in report['optional']:print('Optional:', message)
        print('API dependencies:',report['api_dependencies'])
        for name,location in report['output_locations'].items():print(f"Output {name}: {location['path']} · writable={location['writable']}")
        for message in report['errors']:print('ERROR:', message)
    elif command == 'models':
        for row in report:print(f"{row.get('run_name') or row.get('run_id') or 'linked'} · {row['specialist']} {row['grid']} · update{row['update']} · {row['export_kind']} · {row['status']}\n  {row['path']}")
        if not report:print('No loadable exports yet. Train a preset or pdeflow models --import PATH')
    elif command == 'evaluate':
        print(f"{report['run_id']} · {report['split']} · {report['case_count']} cases · {report['state']}\n{report['purpose']}\n{report['output']}")
        for case, modes in report['reports'].items():
            for mode, metrics in modes.items():print(f'case{case} {mode}: {metrics}')
        print('Full per-field metrics, costs, stop reasons and baseline diagnostics are saved for every case/mode.')
    else:
        for key, value in report.items():print(f'{key}: {value}')
