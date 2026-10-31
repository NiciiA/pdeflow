"""Persistent user training workflows over the existing scientific trainers.

Workers survive browser/API disconnects. Kernel locks own execution; manifests
describe it. This module contains no optimizer or model training loop.
"""
from __future__ import annotations

from dataclasses import replace
from importlib.resources import files
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from pdeflow import local
from pdeflow.runtime import Config, RuntimeSpec, code_identity, digest, file_digest, read_json, write_json

PROBLEMS = ('wave-2d','heat-wave-2d','heat-1d', 'heat-2d', 'heat-3d', 'darcy-2d', 'darcy-3d', 'chemistry-2d', 'chemistry-3d')
MODELS = {'resolution-flexible': 'Operator-Aware Generative Latent Neural Operator'}
BUDGETS = {
    'pipeline-check': 'Verifies execution; unsuitable for judging accuracy.',
    'development': 'Bounded experiment for assessing learning; adequacy unestablished.',
    'extended': 'Larger configurable budget; resource fit and accuracy are not guaranteed.',
}
STAGE_LABELS = {
    'prepare': ('Prepare examples and fixed splits', 'Independent physical realizations; validation and test remain separate.'),
    'representation': ('Learn field representation', 'Compress complete training fields and decode physical coordinates (autoencoder).'),
    'representation_validation': ('Check field representation', 'Held-out validation distortion compared with a training-fitted constant predictor.'),
    'evolution': ('Learn physical-time evolution','Frozen state encoder/decoder; shared graph and transition backbone learn complete physical-state dynamics.'),
    'flow': ('Learn conditional generation', 'Learn latent velocity conditioned on noisy observations; field representation stays frozen.'),
    'fixed': ('Learn fixed-grid generation', 'Learn field-space flow velocity; observations guide reconstruction afterward.'),
    'model_validation': ('Inspect validation reconstructions', 'Fixed held-out validation cases and sensor seeds; never the test split.'),
    'register': ('Make model available', 'Discoverable inference export; use requires your explicit selection.'),
}
ACTIVE = ('queued', 'running', 'stopping')


def known_workspaces(workspace=None):
    """Discover manifests within project outputs only; never move historical data.

    Isolated external workspaces stay isolated. Dataset/cache trees are pruned.
    Explicit workspace links support outputs outside the source checkout.
    """
    root = local.workspace_root(workspace)
    result = {root}
    outputs = local.project_root() / 'outputs'
    if root.is_relative_to(outputs):
        for directory, dirs, _ in os.walk(outputs, followlinks=False):
            p = Path(directory)
            if 'runs' in dirs or 'coordinate' in dirs or 'training-workflows' in dirs:
                result.add(p.resolve())
            dirs[:] = [d for d in dirs if d not in {
                'runs', 'coordinate', 'training-workflows', 'data', 'cache', 'workbench',
                'evaluation', 'evaluations', 'datasets', 'node_modules', '.storage',
            } and not (p / d).is_symlink()]
    links = root / 'workspace-links.json'
    if links.exists():
        result.update(Path(p).expanduser().resolve() for p in read_json(links)['paths'])
    return sorted(result, key=lambda p: (p != root, str(p)))


def fixed_model_entries(workspace=None):
    return [entry for root in known_workspaces(workspace) for entry in local.model_entries(root)]


def options():
    available = {'cpu': True, 'mps': torch.backends.mps.is_available(), 'cuda': torch.cuda.is_available()}
    return {'problems': ['heat-2d','wave-2d','heat-wave-2d'], 'models': MODELS, 'budgets': BUDGETS,
            'devices': available, 'auto_device': 'cuda' if available['cuda'] else 'mps' if available['mps'] else 'cpu',
            'auto_explanation': 'Auto selects NVIDIA GPU when available, then Apple GPU, then CPU. Explicit unavailable choices reject; no concealed fallback.'}


def _coordinate_profile(problem, budget):
    from pdeflow.colab import training_preset as preset
    from dataclasses import replace
    c=preset('smoke' if budget=='pipeline-check' else 'xsmall' if budget=='development' else 'small',device='cpu',precision_policy='fp32')
    weights={'heat':.5,'wave':.5} if problem=='heat-wave-2d' else {problem.split('-')[0]:1.}
    if budget=='development':c=replace(c,train_count=64,validation_count=12,test_count=12,ae_steps=500,evolution_steps=500,flow_steps=500,checkpoint_every=50)
    return replace(c,family_weights=weights).validate().to_dict()


def plan(request):
    allowed = {'problem', 'model', 'budget', 'device', 'advanced', 'allow_poor_representation', 'representation_max_mse_ratio'}
    if set(request) - allowed:
        raise ValueError('Unknown training settings: ' + ', '.join(sorted(set(request) - allowed)))
    request = {'problem': 'heat-wave-2d', 'model': 'resolution-flexible', 'budget': 'pipeline-check', 'device': 'auto',
               'advanced': {}, 'allow_poor_representation': False, 'representation_max_mse_ratio': 1., **request}
    problem, model, budget = (request[k] for k in ('problem', 'model', 'budget'))
    if problem not in PROBLEMS or model not in MODELS or budget not in BUDGETS:
        raise ValueError('Choose a supported problem, model and budget preset')
    if model == 'resolution-flexible' and problem not in ('heat-2d','wave-2d','heat-wave-2d'):
        raise ValueError('The shared operator supports periodic heat/wave2D')
    dev, _ = RuntimeSpec(device=request['device']).resolve()
    advanced = request.get('advanced') or {}
    if not isinstance(advanced, dict):
        raise ValueError('Advanced settings must be named configuration sections')
    if type(request.get('allow_poor_representation', False)) != bool:
        raise ValueError('Representation override must be explicit true/false')
    ratio = request.get('representation_max_mse_ratio', 1.)
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('Representation gate ratio must be in (0,1]; 1 means better than the constant validation baseline')
    request['advanced'] = advanced
    request['representation_max_mse_ratio'] = float(ratio)
    from pdeflow.data.physical import distribution
    if model == 'resolution-flexible':
        from pdeflow.coordinate.contracts import CoordinateConfig
        from pdeflow.coordinate.training import data_config
        config = _coordinate_profile(problem, budget)
        allowed_advanced = set(CoordinateConfig.__dataclass_fields__) - {'dimension', 'runtime'}
        if set(advanced) - allowed_advanced:
            raise ValueError('Unsupported coordinate advanced setting')
        if 'field_loss' in advanced:advanced={**advanced,'field_loss':{**config.get('field_loss',{}),**advanced['field_loss']}}
        config.update(advanced)
        config['runtime'] = {**config['runtime'], 'device': str(dev)}
        c = CoordinateConfig(**config).validate()
        expected={'heat','wave'} if problem=='heat-wave-2d' else {problem.split('-')[0]}
        if set(c.family_weights)!=expected:raise ValueError('Training equations conflict with advanced family_weights; choose matching Heat only, Wave only or Heat + Wave. No family fallback is applied.')
        from pdeflow.coordinate.physics import training_spec
        physics=training_spec(c)
        physical = data_config(c)
        updates = {'representation': c.ae_steps,'evolution':c.evolution_steps,'flow': c.flow_steps}
        examples = {'training': c.train_count, 'validation': c.validation_count, 'untouched_test': c.test_count}
        config = c.to_dict()
        stages = ['prepare', 'representation', 'representation_validation','evolution','flow', 'model_validation', 'register']
    # A custom update count must not retain the pipeline-check label accidentally.
    pipeline = budget == 'pipeline-check'
    return {'schema_version': 1, 'request': request, 'problem': problem, 'model': model, 'model_label': MODELS[model],
            'budget': budget, 'budget_meaning': BUDGETS[budget], 'pipeline_check': pipeline,
            'requested_device': request['device'], 'resolved_device': str(dev), 'config': config,
            'distribution': __import__('pdeflow.coordinate.dataset',fromlist=['recipe']).recipe(c) if model=='resolution-flexible' and c.temporal_mode=='continuous' else distribution(physical), 'examples': examples, 'updates': updates,
            'architecture_parameters':__import__('pdeflow.coordinate.presets',fromlist=['counts']).counts(c),
            'batch_settings':__import__('pdeflow.training_settings',fromlist=['resolve_batches']).resolve_batches(c),'stages': stages,'training_physics':physics, 'grid': physical.to_dict()['grid'],
            'equation': ' · '.join(physics['equations'].values()),
            'varies': 'Periodic initial-field shape mixtures, continuous physical times and an independent observation-placement mixture. Initial-field mixture types are not PDE families.',
            'quality_policy': {'max_mse_ratio': float(ratio), 'override': request.get('allow_poor_representation', False),
                               'pipeline_exemption': pipeline,
                               'meaning': 'Validation MSE must improve on a training-fitted constant predictor. This is a minimal sanity check, not a universal adequacy threshold. Pipeline checks exercise stages without claiming quality.'}}


def _root(workspace=None):
    return local.initialize(workspace) / 'training-workflows'


def _resolve(selector='latest', workspace=None):
    root = _root(workspace)
    paths = [file for directory in known_workspaces(workspace) for file in (directory / 'training-workflows').glob('*/workflow.json')]
    if selector == 'latest':
        paths = sorted(paths, key=lambda p: read_json(p)['created_at'], reverse=True)
        if not paths:
            raise ValueError('No integrated training run yet; start with pdeflow training start')
        return paths[0].parent
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', selector):
        raise ValueError('Use the recorded training run ID, not a checkpoint path')
    path = next((p.parent for p in paths if p.parent.name == selector), root / selector)
    if not (path / 'workflow.json').is_file():
        raise ValueError('Unknown integrated training run')
    return path


def _save(path, record, **changes):
    record.update(changes, updated_at=local.now())
    write_json(path / 'workflow.json', record)


def _spawn(path):
    with (path / 'worker.log').open('a') as log:
        worker = subprocess.Popen([sys.executable, '-m', 'pdeflow.training_workflow', str(path)],
                                  stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, close_fds=True)
    return worker.pid


def start(request, workspace=None, token=None, background=True):
    from .batch_experiments import status as batch_status,ACTIVE as batch_active
    for file in (local.workspace_root(workspace)/'batch-experiments').glob('*/experiment.json'):
        if batch_status(file.parent)['state'] in batch_active:raise ValueError('Finish/cancel the batch benchmark before training')
    resolved = plan(request)  # validate availability/physics before any run allocation
    root = _root(workspace); root.mkdir(parents=True, exist_ok=True)
    token = token or uuid.uuid4().hex
    if not isinstance(token, str) or not 1 <= len(token) <= 100:
        raise ValueError('Submission token must have 1..100 characters')
    with local.run_lock(root):
        for file in root.glob('*/workflow.json'):
            old = read_json(file)
            if old['submission_token'] == token:
                if digest(old['plan']['request']) != digest(resolved['request']):
                    raise ValueError('Submission token already identifies different settings')
                return status(file.parent.name, workspace)
            if digest(old['plan']['request']) == digest(resolved['request']) and status(file.parent.name, workspace)['state'] in ACTIVE:
                return status(file.parent.name, workspace)
        identifier = time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '-' + uuid.uuid4().hex[:10]
        path = root / identifier; path.mkdir()
        record = {'schema_version': 1, 'kind': 'pdeflow_training_workflow', 'id': identifier,
                  'workspace': str(local.workspace_root(workspace)), 'submission_token': token,
                  'plan': resolved, 'state': 'queued', 'stage': 'prepare', 'finished_stages': [],
                  'created_at': local.now(), 'updated_at': local.now(), 'source': code_identity(),
                  'pid': None, 'validation': None, 'model_id': None, 'attempts': 1}
        write_json(path / 'resolved-config.json', resolved['config']); _save(path, record)
        if background:
            _save(path, record, pid=_spawn(path), launched_unix=time.time())
    if not background:
        execute(path)
    return status(identifier, workspace)


def _child_progress(path, record):
    p = record['plan']
    directory = path / ('evolution' if record['stage']=='evolution' else 'coordinate-ae' if record['stage'] in ('prepare', 'representation', 'representation_validation') else 'coordinate-flow')
    if p['model'] == 'fixed-grid':
        directory = path / 'fixed'
    file = directory / 'run.json'
    if not file.exists():
        return {}, []
    child = read_json(file)
    history_file = directory / ('training/history.json' if p['model'] == 'fixed-grid' else 'history.json')
    history = read_json(history_file) if history_file.exists() else []
    if child.get('history') and (not history or child['history'][-1]['step'] > history[-1]['step']):
        history = child['history']
    if p['model'] == 'fixed-grid' and (directory / 'progress.json').exists():
        live = read_json(directory / 'progress.json')
        child['progress'] = max(child['progress'], live.get('step', 0))
        child['eta_seconds'] = live.get('eta_seconds')
        child['training_examples_consumed'] = live.get('unique_physical_problems')
        if live.get('latest') and (not history or history[-1]['step'] != live['latest']['step']):
            history = [*history, live['latest']]
    accounting_path = directory / 'training/data-accounting.json'
    if p['model'] == 'fixed-grid' and accounting_path.exists():
        accounting = read_json(accounting_path)
        child['training_examples_consumed'] = max(child.get('training_examples_consumed') or 0, len(set(accounting.get('consumed_ids', []))))
    return child, history[-200:]


def status(selector='latest', workspace=None):
    path = _resolve(selector, workspace); record = read_json(path / 'workflow.json')
    active = local.is_locked(path)
    state = record['state']
    if state in ACTIVE and not active and time.time() - record.get('launched_unix', 0) > 10:
        state = 'interrupted'
    child, history = _child_progress(path, record)
    stage = 'prepare' if child.get('state') == 'preparing' and active else record['stage']
    progress = child.get('progress', 0) if record['stage'] in ('prepare','representation','evolution','flow','fixed') else 0
    total = record['plan']['updates'].get(record['stage'], 0)
    elapsed = record.get('elapsed_seconds_total', 0.)
    if active:
        elapsed += time.time() - record.get('attempt_started_unix', time.time())
    eta = child.get('eta_seconds') if active else None
    # Resume resets stage clocks. Use only a monotone current-session window.
    timed = []
    for row in history:
        if 'elapsed_seconds' not in row:continue
        if timed and row['elapsed_seconds'] < timed[-1]['elapsed_seconds']:timed=[]
        timed.append(row)
    if active and len(timed) >= 21:
        seconds = timed[-1]['elapsed_seconds'] - timed[0]['elapsed_seconds']
        updates = timed[-1]['step'] - timed[0]['step']
        if seconds >= 10 and updates >= 20:
            eta = max(0, total - progress) * seconds / updates
    training_seconds = child.get('training_seconds')
    return {**record, 'state': state, 'active': active, 'stage': stage, 'stage_label': STAGE_LABELS[stage][0],
            'stage_meaning': STAGE_LABELS[stage][1], 'progress': progress, 'stage_budget': total,
            'history': history, 'elapsed_seconds': max(0., elapsed), 'eta_seconds': eta,
            'eta_scope': 'Current learning stage only; preparation/evaluation excluded. Available after >=20 timed updates and >=10 seconds.',
            'child_state': child.get('state'), 'training_seconds': training_seconds,
            'training_examples_consumed': child.get('training_examples_consumed'),
            'stop_requested': (path / 'stop-request.json').exists(),
            'log_tail': (path / 'worker.log').read_text(errors='replace')[-16000:] if (path / 'worker.log').exists() else '',
            'stage_log_tail': (path / 'fixed/train.log').read_text(errors='replace')[-8000:] if (path / 'fixed/train.log').exists() else '',
            'can_resume': state in ('interrupted', 'failed', 'needs_review') and not active,
            'meaning': 'Pipeline check; not evidence of accuracy.' if record['plan']['pipeline_check'] else 'Completed budget and losses do not establish reconstruction accuracy.'}


def runs(workspace=None):
    _root(workspace)
    paths = [p for root in known_workspaces(workspace) for p in (root/'training-workflows').glob('*/workflow.json')]
    return [status(p.parent.name, workspace) for p in sorted(paths, key=lambda p:read_json(p)['created_at'], reverse=True)]


def stop(selector='latest', workspace=None):
    path = _resolve(selector, workspace)
    # Request files are cooperative; never signal a PID that may have been reused.
    with local.run_lock(path.parent):
        record = read_json(path / 'workflow.json')
        current = status(path.name, workspace)
        if current['state'] not in ACTIVE:
            return current
        write_json(path / 'stop-request.json', {'requested_at': local.now()})
    return status(path.name, workspace)


def resume(selector='latest', workspace=None, allow_poor_representation=False, background=True):
    path = _resolve(selector, workspace)
    with local.run_lock(path.parent):
        current = status(path.name, workspace)
        if current['active'] or current['state'] in ACTIVE:
            return current
        if current['state'] == 'completed':
            raise ValueError('This budget is complete. Start a new experiment; latent train-further with changed targets/configuration is not supported.')
        record = read_json(path / 'workflow.json')
        if record['plan']['model']!='resolution-flexible':raise ValueError('Legacy generative training is retired; start a new joint model run. Existing artifacts remain untouched.')
        if record['source']['source_sha256'] != code_identity()['source_sha256']:
            raise ValueError('Resume requires the recorded source version. Inference exports remain usable; start a new run after code changes.')
        if allow_poor_representation:
            record['quality_override'] = {'at': local.now(), 'reason': 'Explicit user continuation after inspecting representation validation'}
        (path / 'stop-request.json').unlink(missing_ok=True)
        _save(path, record, state='queued', error=None, attempts=record['attempts'] + 1,
              launched_unix=time.time())
        if background:
            _save(path, record, pid=_spawn(path))
    if not background:
        execute(path)
    return status(path.name, workspace)


def extend(selector='latest', additional_updates=100, workspace=None, background=True):
    """Explicit fixed-grid continuation, preserving the previous budget snapshot."""
    import shutil
    path = _resolve(selector, workspace)
    if type(additional_updates) != int or not 1 <= additional_updates <= 1_000_000:
        raise ValueError('Additional updates must be an integer in 1..1000000')
    with local.run_lock(path.parent):
        current = status(path.name, workspace)
        if current['active'] or current['state'] != 'completed':
            raise ValueError('Train further requires an inactive completed budget; resume an interrupted run instead')
        record = read_json(path / 'workflow.json')
        if record['plan']['model'] != 'fixed-grid':
            raise ValueError('Changing latent stage budgets is not supported by strict resume; start a new representation/flow experiment')
        if record['source']['source_sha256'] != code_identity()['source_sha256']:
            raise ValueError('Use the recorded source for continuation; start a new run after source changes')
        old_budget = record['plan']['updates']['fixed']
        snapshot = path / 'budgets' / str(old_budget)
        snapshot.mkdir(parents=True, exist_ok=False)
        for name in ('checkpoint.pt', 'inference.pt', 'best-inference.pt', 'history.json'):
            original = path / 'fixed/training' / name
            if original.is_file():shutil.copy2(original, snapshot / name)
        write_json(snapshot / 'workflow.json', record)
        # Preserve the completed model as a selectable export when inference.pt
        # advances atomically. Registration only adds a link to our snapshot.
        local.models(record['workspace'], import_path=snapshot / 'inference.pt')
        config = record['plan']['config']
        advanced = {k:v for k,v in config.items() if k in ('grid','data','model','training','provider','inference','chemistry') and v is not None}
        advanced['training'] = {**advanced['training'], 'steps': old_budget + additional_updates}
        resolved = plan({'problem':record['plan']['problem'], 'model':'fixed-grid', 'budget':'development',
                         'device':record['plan']['resolved_device'], 'advanced':advanced})
        write_json(path / f'resolved-config-{old_budget + additional_updates}.json', resolved['config'])
        _save(path, record, plan=resolved, state='queued', stage='fixed', finished_stages=['prepare'],
              validation=None, model_id=None, export=None, error=None,
              validation_report_file=f'model-validation-{old_budget + additional_updates}.json',
              continuations=[*record.get('continuations', []), {'from_updates':old_budget,'additional_updates':additional_updates,
                             'snapshot':str(snapshot),'at':local.now()}],
              attempts=record['attempts']+1, launched_unix=time.time())
        (path / 'stop-request.json').unlink(missing_ok=True)
        if background:_save(path, record, pid=_spawn(path))
    if not background:execute(path)
    return status(path.name, workspace)


def representation_quality(ae_path, report, max_ratio=1.):
    """Compare held-out distortion to a constant fitted WITHOUT validation data."""
    from pdeflow.coordinate.training import read_checkpoint
    cp, _ = read_checkpoint(ae_path, 'autoencoder')
    metadata = read_json(Path(ae_path).parent / 'dataset.json')
    if 'constant_baseline_mse' in report:baseline=report['constant_baseline_mse']
    else:
        with np.load(Path(ae_path).parent / 'dataset.npz', allow_pickle=False) as data:
            values = data['values'][[i for i, split in enumerate(metadata['splits']) if split == 'validation']]
        baseline = float(np.mean((values - cp['normalization']['mean']) ** 2))
    mse = float(report['validation_mse']) if 'validation_mse' in report else float(np.mean([r['mse_per_sample'][0] for r in report['records']]))
    ratio = mse / baseline if baseline > 1e-15 else None
    passed = bool(math.isfinite(mse) and ratio is not None and ratio < max_ratio)
    return {'passed': passed, 'validation_mse': mse, 'constant_baseline_mse': baseline, 'mse_ratio': ratio,
            'max_mse_ratio': max_ratio, 'baseline': 'Constant training-field mean; evaluated only on validation fields',
            'meaning': 'A minimal baseline check, not proof of an adequate representation. Inspect absolute, initial-condition and spectral errors.'}


def _coordinate_validation(path, config, device):
    from pdeflow.coordinate.validation import flow
    return flow(path/'coordinate-flow/coordinate.pt',path/'flow-validation.json',device,subset=2,steps=2 if config['flow_steps']<=4 else 8,particles=1)


def execute(path):
    """One process owns an entire workflow; every scientific stage is reused."""
    path = Path(path).resolve()
    with local.run_lock(path):
        record = read_json(path / 'workflow.json'); p = record['plan']
        interrupted = [False]
        previous = signal.signal(signal.SIGINT, lambda *_: interrupted.__setitem__(0, True))
        def should_stop():
            return interrupted[0] or (path / 'stop-request.json').exists()
        def stage(key):
            if should_stop():
                raise InterruptedError('Safe stop requested; completed stage artifacts and resume state preserved')
            _save(path, record, stage=key, state='running')
            print(STAGE_LABELS[key][0], flush=True)
        def finish(key):
            if key not in record['finished_stages']:
                _save(path, record, finished_stages=[*record['finished_stages'], key])
        attempt_started = time.time()
        _save(path, record, pid=os.getpid(), state='running', started_unix=record.get('started_unix', attempt_started),
              attempt_started_unix=attempt_started, error=None)
        try:
            if p['model'] == 'resolution-flexible':
                outcome=coordinate_stages(p['config'],path,mode='resume' if (path/'coordinate-ae/run.json').exists() else 'new',
                    gate_ratio=p['quality_policy']['max_mse_ratio'],override_gate=bool(p['pipeline_check'] or p['quality_policy']['override'] or record.get('quality_override')),
                    names=('coordinate-ae','coordinate-flow'),stage_hook=stage,finish_hook=finish,should_stop=should_stop,local_validation=True)
                _save(path,record,representation_validation=outcome.get('representation_validation'),representation_quality=outcome.get('quality'),
                    gate_decision='pipeline execution exemption' if p['pipeline_check'] else 'explicit override' if outcome.get('gate_override') else 'baseline check passed')
                if outcome['state']=='needs_review':
                    _save(path,record,state='needs_review',finished_unix=time.time(),warning='Representation did not beat the constant validation baseline. Inspect it before explicitly overriding the gate.')
                    return
                if outcome['state']!='completed':raise InterruptedError('Coordinate training stopped at a saved optimizer boundary')
                validation=outcome['validation'];export=Path(outcome['export'])
            else:raise ValueError('Legacy generative workflow is retired; start a new joint model run')
            _save(path, record, validation=validation)
            finish('model_validation'); stage('register')
            if p['model'] == 'resolution-flexible':
                from pdeflow.coordinate.registry import register_checkpoint
                register_checkpoint(export, record['workspace'])
            # Registration never changes the user's selected reconstruction model.
            finish('register')
            _save(path, record, state='completed', model_id=file_digest(export)[:32], export=str(export), finished_unix=time.time(),
                  warning='Pipeline-check checkpoint: execution only; unsuitable for judging accuracy.' if p['pipeline_check'] else 'Inspect held-out validation; completion does not establish useful accuracy.')
        except (InterruptedError, KeyboardInterrupt) as error:
            _save(path, record, state='interrupted', error=str(error), finished_unix=time.time())
        except Exception as error:
            import traceback
            traceback.print_exc()
            _save(path, record, state='failed', error=f'{type(error).__name__}: {error}', finished_unix=time.time())
        finally:
            _save(path, record, elapsed_seconds_total=record.get('elapsed_seconds_total', 0.) + time.time() - attempt_started)
            signal.signal(signal.SIGINT, previous)


def model_list(workspace=None, problem=None):
    from pdeflow.coordinate.registry import runs as coordinate_runs
    from pdeflow.coordinate.contracts import CoordinateConfig
    from pdeflow.coordinate.training import data_config
    from pdeflow.core import ProblemSpec
    from pdeflow.inference import compatibility_report
    from pdeflow.specialists import get_specialist
    from pdeflow.training import _read_checkpoint
    p = ProblemSpec.from_dict(problem) if problem else None
    workflows = {}
    for root in known_workspaces(workspace):
        for file in (root / 'training-workflows').glob('*/workflow.json'):
            row = read_json(file)
            if row.get('model_id'):
                workflows[row['model_id']] = row
            for snapshot in file.parent.glob('budgets/*/workflow.json'):
                older = {**read_json(snapshot), 'is_snapshot':True}
                if older.get('model_id'):workflows[older['model_id']] = older
    result = {}
    for entry in coordinate_runs(workspace):
        if entry.get('stage') != 'flow' or not entry.get('available'):
            continue
        c = CoordinateConfig(**entry['config']).validate()
        from pdeflow.coordinate.physics import problem as dynamic_problem
        family=p.family if p and p.family in c.family_weights else next(iter(c.family_weights));bounds=c.diffusivity_range if family=="heat" else c.wave_speed_range
        physical=dynamic_problem(data_config(c).grid,family,sum(bounds)/2)
        compatible, reason = False, 'No current executable problem; use the model training problem.'
        if p:
            try:
                c.compatible(p); compatible, reason = True, 'Physical contract matches; query resolution may vary.'
            except ValueError as error:
                reason = str(error)
        identifier = entry['model_id']
        result[identifier] = {'id': identifier, 'model': 'resolution-flexible', 'type': MODELS['resolution-flexible'],
                              'problem': physical.to_dict(), 'updates': {'representation':c.ae_steps,'evolution':c.evolution_steps,'flow': entry['progress']},
                              'status': entry['state'], 'path': entry['export'], 'compatibility': {'compatible': compatible, 'summary': reason}, 'available': True}
    for identifier, row in result.items():
        workflow = workflows.get(identifier)
        row.update(workflow_id=workflow['id'] if workflow else None,
                   can_extend=bool(workflow and not workflow.get('is_snapshot') and row['model']=='fixed-grid' and workflow['state']=='completed'),
                   pipeline_check=workflow['plan']['pipeline_check'] if workflow else max(row['updates'].values()) <= 20,
                   validation=workflow.get('validation') if workflow else None,
                   representation_quality=workflow.get('representation_quality') if workflow else None,
                   budget=workflow['plan']['budget'] if workflow else 'historical budget; inspect provenance',
                   quality='Execution only; unsuitable for judging accuracy' if (workflow and workflow['plan']['pipeline_check']) or max(row['updates'].values()) <= 20 else 'Scientifically unevaluated; inspect validation')
    return list(result.values())


def select_model(identifier, workspace=None, problem=None, adopt_training_problem=False):
    from pdeflow.core import ProblemSpec
    from pdeflow.reference.examples import from_problem
    entry = next((r for r in model_list(workspace, problem) if r['id'] == identifier), None)
    if not entry:
        # Integrated run IDs are accepted by the ordinary CLI; paths are internal.
        record = status(identifier, workspace)
        if not record.get('model_id'):
            raise ValueError('No inference artifact is available for this run yet')
        return select_model(record['model_id'], workspace, problem, adopt_training_problem)
    if problem and not entry['compatibility']['compatible'] and not adopt_training_problem:
        raise ValueError(entry['compatibility']['summary'])
    request = from_problem(ProblemSpec.from_dict(entry['problem']), seed=53).request if entry['problem']['family']=='heat' else None
    selection = {'id': entry['id'], 'model': entry['model'], 'selected_at': local.now(),
                 'pipeline_check': entry['pipeline_check'], 'problem': entry['problem'],
                 'suggested_request': request, 'adopt_training_problem': adopt_training_problem,
                 'meaning': entry['quality']}
    root = local.initialize(workspace)
    write_json(root / 'selected-model.json', selection)
    return selection


def preview(selector='latest', workspace=None, case=0, field=0, time_index=0, axis=2, index=0):
    from pdeflow.core import ProblemSpec
    from pdeflow.reference import slice_payload
    path = _resolve(selector, workspace); record = read_json(path / 'workflow.json')
    if type(case) != int or not 0 <= case < 2:
        raise ValueError('Choose a recorded validation case 0 or 1')
    if record['plan']['model'] == 'resolution-flexible':
        from pdeflow.coordinate.physics import problem as dynamic_problem
        from pdeflow.core import GridSpec,DomainSpec
        report=read_json(path/'flow-validation.json');rows=report['records'];row=rows[min(case*2,len(rows)-1)];offset=0
        for r in rows[:min(case*2,len(rows)-1)]:offset+=2 if r['family']=='wave' else 1
        with np.load(path/'flow-validation.npz',allow_pickle=False) as data:
            available=data['preview'];count=2 if row['family']=='wave' else 1
            if offset+count>len(available):raise ValueError('Selected bounded validation preview unavailable')
            selected=available[offset:offset+count];reference=selected[:,0];predicted=selected[:,1]
        cfg=record['plan']['config'];grid=GridSpec(tuple(predicted.shape[-2:]),DomainSpec(tuple(cfg['lengths']),tuple(cfg['origins'])),tuple(report['settings']['times']));problem=dynamic_problem(grid,row['family'],None)

    else:
        report = read_json(path / record.get('validation_report_file', 'model-validation.json')); evaluation = Path(report['output'])
        cases = sorted((evaluation).glob('*/observation_pde/result.npz'))
        if case >= len(cases):
            raise ValueError('Validation case unavailable')
        directory = cases[case].parent
        problem = ProblemSpec.from_dict(read_json(directory / 'problem.json'))
        with np.load(cases[case], allow_pickle=False) as data:
            predicted = data['samples'].mean(0); reference = data['reference'][0] if data['reference'].ndim == predicted.ndim + 1 else data['reference']
    if not 0 <= field < len(problem.fields) or not 0 <= time_index < len(problem.grid.times):
        raise ValueError('Field/physical-time index outside validation case')
    a, b = predicted[field, time_index], reference[field, time_index]
    payload = slice_payload(problem, a, problem.fields[field].name, axis, index, time_index)
    payload['reference'] = slice_payload(problem, b, problem.fields[field].name, axis, index, time_index)['values']
    payload['error'] = np.abs(np.asarray(payload['values']) - payload['reference']).tolist()
    payload['vmin'] = min(payload['vmin'], float(np.min(payload['reference'])))
    payload['vmax'] = max(payload['vmax'], float(np.max(payload['reference'])))
    return payload


def coordinate_stages(config,path,stages='both',mode='new',dependency=None,gate_ratio=1.,override_gate=False,stop_after=None,event=None,names=('autoencoder','flow'),stage_hook=None,finish_hook=None,should_stop=None,local_validation=False):
    """One shared A1 → A2 → B local/Colab driver, immutable stage dependencies."""
    from pdeflow.coordinate.contracts import CoordinateConfig
    from pdeflow.coordinate.training import train,read_checkpoint
    from pdeflow.coordinate.validation import representation,evolution as evaluate_evolution,flow as evaluate_flow
    from pdeflow.coordinate.optimization import resume_contract
    c=CoordinateConfig(**config).validate();path=Path(path);ae=path/names[0];ev=path/'evolution';fl=path/names[1]
    if stages not in ('both','representation','evolution','flow'):raise ValueError('Choose both, representation, evolution or flow')
    def stage(key):
        if stage_hook:stage_hook(key)
    def finish(key):
        if finish_hook:finish_hook(key)
    def run(kind,root,dep=None):
        stage({'autoencoder':'representation'}.get(kind,kind));resume=(root/'resume.pt').exists()
        if resume:
            cp,old_c=read_checkpoint(root/'resume.pt',kind)
            from .coordinate.optimization import model_contract,scientific_source
            if cp['completed'] and cp['progress']==getattr(c,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[kind]) and model_contract(old_c)==model_contract(c) and cp['scientific_source']['sha256']==scientific_source()['sha256']:return read_json(root/'run.json')
            resume_contract(cp,c,kind,mode=='extend')
            total=getattr(c,{'autoencoder':'ae_steps','evolution':'evolution_steps','flow':'flow_steps'}[kind])
            if cp['completed'] and cp['progress']==total:return read_json(root/'run.json')
            if kind!='flow' and (fl/'resume.pt').exists():raise ValueError('Changing the frozen representation/operator requires a new downstream run')
        return train(config,root,kind,dep,resume,stop_after,should_stop=should_stop or (lambda:(path/'stop-request.json').exists()),event=event,extend=mode=='extend')
    if stages in ('both','representation'):
        r=run('autoencoder',ae)
        if r['state']!='completed':return {'state':'interrupted','stage':'autoencoder','progress':r['progress']}
        finish('prepare');finish('representation')
    aep=ae/'autoencoder.pt';evp=ev/'evolution.pt'
    if dependency:
        dcp,_=read_checkpoint(dependency)
        if dcp['stage']=='autoencoder':aep=Path(dependency)
        elif dcp['stage']=='evolution':
            evp=Path(dependency);aep=(evp.parent/dcp['dependency']['path']).resolve()
        else:raise ValueError('Frozen dependency must be A1 or A2')
    if not aep.exists():raise ValueError('Choose a compatible completed A1 representation')
    stage('representation_validation');report_path=path/'representation-validation.json'
    if report_path.exists() and read_json(report_path)['checkpoint_sha256']==file_digest(aep):report=read_json(report_path)
    else:
        if report_path.exists():report_path.rename(path/('representation-validation-'+uuid.uuid4().hex[:8]+'.json'))
        report=representation(aep,report_path,c.runtime['device'],subset=min(c.validation_count,12))
    quality=representation_quality(aep,report,gate_ratio);write_json(path/'representation-quality.json',quality);finish('representation_validation')
    if stages=='representation':return {'state':'completed','stage':'representation_validation','quality':quality}
    if not quality['passed'] and not override_gate:return {'state':'needs_review','stage':'representation_gate','quality':quality,'representation_validation':report}
    # Even flow-only runs need an actual trained A2 dependency. No decoder shortcut.
    if not dependency or dcp['stage']!='evolution':
        r=run('evolution',ev,aep)
        if r['state']!='completed':return {'state':'interrupted','stage':'evolution','progress':r['progress'],'quality':quality}
    finish('evolution')
    er=path/'evolution-validation.json'
    if not er.exists():evaluate_evolution(evp,er,c.runtime['device'],subset=min(c.validation_count,12))
    if stages=='evolution':return {'state':'completed','stage':'evolution','quality':quality}
    r=run('flow',fl,evp)
    if r['state']!='completed':return {'state':'interrupted','stage':'flow','progress':r['progress'],'quality':quality}
    finish('flow');stage('model_validation');report_path=path/'flow-validation.json'
    if report_path.exists() and read_json(report_path)['checkpoint_sha256']==file_digest(fl/'coordinate.pt'):evaluation=read_json(report_path)
    else:
        if report_path.exists():report_path.rename(path/('flow-validation-'+uuid.uuid4().hex[:8]+'.json'))
        evaluation=evaluate_flow(fl/'coordinate.pt',report_path,c.runtime['device'],subset=2 if c.flow_steps<=4 else min(c.validation_count,12),particles=1,steps=2 if c.flow_steps<=4 else 8)
    return {'state':'completed','stage':'evaluated','quality':quality,'representation_validation':report,'gate_override':override_gate,'validation':evaluation,'export':str(fl/'coordinate.pt')}


if __name__ == '__main__':
    execute(Path(sys.argv[1]))
