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

PROBLEMS = ('heat-1d', 'heat-2d', 'heat-3d', 'darcy-2d', 'darcy-3d', 'chemistry-2d', 'chemistry-3d')
MODELS = {'fixed-grid': 'Existing fixed-grid model', 'resolution-flexible': 'Resolution-flexible model'}
BUDGETS = {
    'pipeline-check': 'Verifies execution; unsuitable for judging accuracy.',
    'development': 'Bounded experiment for assessing learning; adequacy unestablished.',
    'extended': 'Larger configurable budget; resource fit and accuracy are not guaranteed.',
}
STAGE_LABELS = {
    'prepare': ('Prepare examples and fixed splits', 'Independent physical realizations; validation and test remain separate.'),
    'representation': ('Learn field representation', 'Compress complete training fields and decode physical coordinates (autoencoder).'),
    'representation_validation': ('Check field representation', 'Held-out validation distortion compared with a training-fitted constant predictor.'),
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
    return {'problems': list(PROBLEMS), 'models': MODELS, 'budgets': BUDGETS,
            'devices': available, 'auto_device': 'cuda' if available['cuda'] else 'mps' if available['mps'] else 'cpu',
            'auto_explanation': 'Auto selects NVIDIA GPU when available, then Apple GPU, then CPU. Explicit unavailable choices reject; no concealed fallback.'}


def _coordinate_profile(problem, budget):
    name = 'heat-full.json' if budget == 'extended' else 'heat-smoke.json'
    c = json.loads(files('pdeflow').joinpath('coordinate', 'profiles', name).read_text())
    dimension = int(problem[-2])
    size = c['reference_shape'][0] if dimension != 3 or budget != 'extended' else 16
    c.update(dimension=dimension, reference_shape=[size] * dimension, lengths=[1.] * dimension, origins=[0.] * dimension)
    if dimension==2:
        from pdeflow.field_losses import recommended
        c['field_loss']=recommended()
        c.update(temporal_mode='continuous',time_interval=[0.,1.],data_mode='procedural',validation_count=max(6,c['validation_count']),test_count=max(6,c['test_count']))
    if budget == 'development':
        c.update(train_count=64, validation_count=8, test_count=8, ae_steps=500, flow_steps=500,
                 encoder_points=256, query_points=256, width=32, checkpoint_every=50)
    return c


def plan(request):
    allowed = {'problem', 'model', 'budget', 'device', 'advanced', 'allow_poor_representation', 'representation_max_mse_ratio'}
    if set(request) - allowed:
        raise ValueError('Unknown training settings: ' + ', '.join(sorted(set(request) - allowed)))
    request = {'problem': 'heat-2d', 'model': 'fixed-grid', 'budget': 'pipeline-check', 'device': 'auto',
               'advanced': {}, 'allow_poor_representation': False, 'representation_max_mse_ratio': 1., **request}
    problem, model, budget = (request[k] for k in ('problem', 'model', 'budget'))
    if problem not in PROBLEMS or model not in MODELS or budget not in BUDGETS:
        raise ValueError('Choose a supported problem, model and budget preset')
    if model == 'resolution-flexible' and not problem.startswith('heat-'):
        raise ValueError('The resolution-flexible specialist currently supports scalar periodic heat only')
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
        physical = data_config(c)
        updates = {'representation': c.ae_steps, 'flow': c.flow_steps}
        examples = {'training': c.train_count, 'validation': c.validation_count, 'untouched_test': c.test_count}
        config = c.to_dict()
        stages = ['prepare', 'representation', 'representation_validation', 'flow', 'model_validation', 'register']
    else:
        base = 'heat-2d' if problem == 'heat-1d' else problem
        c = local.preset_config('quickstart-heat-2d' if problem.startswith('heat') and budget == 'pipeline-check' else base)
        if c.grid.dimension != int(problem[-2]):
            from pdeflow.core import GridSpec, DomainSpec
            dimension = int(problem[-2])
            c = replace(c, grid=GridSpec((c.grid.shape[0],) * dimension, DomainSpec((1.,) * dimension), c.grid.times))
        if budget == 'pipeline-check':
            c = replace(c, grid=replace(c.grid, shape=(4 if problem.endswith('3d') else 8,) * c.grid.dimension),
                        training=replace(c.training, steps=2, batch_size=1, checkpoint_every=1),
                        provider=replace(c.provider, calibration_count=4, validation_count=2, test_count=2),
                        inference=replace(c.inference, steps=2, particles=2, sensors=min(c.inference.sensors, 8)))
        elif budget == 'development':
            c = replace(c, training=replace(c.training, steps=500, checkpoint_every=50))
        config = c.to_dict()
        allowed_sections = {'data', 'model', 'training', 'provider', 'inference', 'grid', 'chemistry'}
        if set(advanced) - allowed_sections:
            raise ValueError('Fixed-grid advanced settings use data/model/training/provider/inference/grid/chemistry sections')
        for key, changes in advanced.items():
            if not isinstance(changes, dict) or not isinstance(config.get(key), dict):
                raise ValueError(f'{key} must be an existing named configuration section')
            config[key].update(changes)
        config['runtime']['device'] = str(dev)
        c = Config.from_dict(config)
        if c.specialist != problem.rsplit('-', 1)[0] or c.grid.dimension != int(problem[-2]):
            raise ValueError('Advanced settings cannot change the selected specialist/dimension')
        physical = c
        updates = {'fixed': c.training.steps}
        training_examples = c.provider.offline_count if c.provider.mode == 'offline' else f'hybrid: reusable pool up to {c.provider.capacity}; actual consumed count recorded during training' if c.provider.mode == 'hybrid' else 'online: new seeded physical problems; actual consumed count recorded during training'
        examples = {'training': training_examples,
                    'calibration': c.provider.calibration_count, 'validation': c.provider.validation_count,
                    'untouched_test': c.provider.test_count}
        stages = ['prepare', 'fixed', 'model_validation', 'register']
    # A custom update count must not retain the pipeline-check label accidentally.
    pipeline = budget == 'pipeline-check'
    return {'schema_version': 1, 'request': request, 'problem': problem, 'model': model, 'model_label': MODELS[model],
            'budget': budget, 'budget_meaning': BUDGETS[budget], 'pipeline_check': pipeline,
            'requested_device': request['device'], 'resolved_device': str(dev), 'config': config,
            'distribution': __import__('pdeflow.coordinate.dataset',fromlist=['recipe']).recipe(c) if model=='resolution-flexible' and c.temporal_mode=='continuous' else distribution(physical), 'examples': examples, 'updates': updates,
            'stages': stages, 'grid': physical.to_dict()['grid'],
            'equation': {'heat': '∂u/∂τ = D Δu', 'darcy': '−div(a ∇u) = f; a > 0; mean(u) = 0 (pressure gauge)',
                         'chemistry': '∂τ(a,b,c) = diffusion + (−r,−r,+r); r = k_f a b − k_r c'}[physical.specialist],
            'varies': 'Six-family periodic initial-field mixture, continuously sampled physical times and configured sensor-placement mixture.' if model=='resolution-flexible' and c.temporal_mode=='continuous' else {'heat': 'Smooth random initial fields and known constant diffusivity within the recorded range.',
                       'darcy': 'Positive random permeability and compatible mean-zero forcing; zero-mean pressure gauge.',
                       'chemistry': 'Positive initial species fields and the recorded diffusion/reaction parameter ranges.'}[physical.specialist],
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
    directory = path / ('coordinate-ae' if record['stage'] in ('prepare', 'representation', 'representation_validation') else 'coordinate-flow')
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
    progress = child.get('progress', 0) if record['stage'] in ('prepare', 'representation', 'flow', 'fixed') else 0
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
    from pdeflow.coordinate.training import data_config
    from pdeflow.coordinate.contracts import CoordinateConfig, grid_queries
    from pdeflow.coordinate.inference import CoordinateBundle
    from pdeflow.coordinate.evaluation import error_summary
    from pdeflow.equations import heat_problem
    c = CoordinateConfig(**config).validate(); bundle = CoordinateBundle(path / 'coordinate-flow/coordinate.pt', device)
    from pdeflow.coordinate.dataset import Dataset
    data = Dataset(c,path/'coordinate-ae',False)
    records = []
    for i in range(min(2,c.validation_count)):
        item=data.get('validation',i);q=grid_queries(data_config(c).grid).numpy() if c.temporal_mode=='continuous' else data.q.numpy();truth=item['solver'].evaluate(q,item['D'],c.interval) if c.temporal_mode=='continuous' else item['values'];rng=np.random.default_rng(c.seed+9100+i)
        ix=rng.choice(len(q),min(8,c.observation_counts[1],len(q)),replace=False);sigma=sum(c.noise_range)/2
        rows=[dict(field='u',**dict(zip(('x','y','z')[:c.dimension],q[k,:c.dimension].tolist())),time=float(q[k,-1]),value=float(truth[k]+rng.normal(0,sigma)),sigma=sigma) for k in ix]
        p=heat_problem(data_config(c).grid,item['D'])
        prediction,metrics=bundle.sample(p,rows,grid_queries(p.grid),steps=2 if config['flow_steps']<=4 else 12,particles=2,seed=c.seed+8100+i)
        from pdeflow.runtime import atomic_write
        destination=path/'validation'/str(len(records));destination.mkdir(parents=True,exist_ok=True)
        atomic_write(destination/'preview.npz',lambda f:np.savez_compressed(f,prediction=prediction.mean(0),reference=truth))
        write_json(destination/'problem.json',p.to_dict());write_json(destination/'observations.json',rows)
        records.append({'id':item['id'],**error_summary(prediction,truth),'observation_consistency':metrics['observation_consistency'],'case':len(records)})
    return {'split': 'validation', 'records': records, 'meaning': 'Fixed validation cases/seeds; two generated samples. No test evaluation or calibrated uncertainty.'}


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
            else:
                child = path / 'fixed'
                if 'fixed' not in record['finished_stages']:
                    stage('fixed')
                    if (child / 'run.json').exists():
                        from pdeflow.training import _read_checkpoint
                        checkpoint = child / 'training/checkpoint.pt'
                        old_config = _read_checkpoint(checkpoint)[1] if checkpoint.exists() else Config.load(child / 'config.json', check_device=False)
                        extra = p['config']['training']['steps'] - old_config.training.steps
                        result = local.resume_training(str(child), additional_steps=extra if extra > 0 else None, should_stop=should_stop)
                    else:
                        result = local.start_training(config_path=path / 'resolved-config.json', workspace=record['workspace'],
                                                      output=child, name=f"{p['problem']} · {p['budget']}", should_stop=should_stop)
                    if result['state'] != 'completed':
                        raise InterruptedError('Fixed-grid generation stopped at a saved update')
                    finish('prepare'); finish('fixed')
                stage('model_validation')
                report_path = path / record.get('validation_report_file', 'model-validation.json')
                validation = read_json(report_path) if report_path.exists() else local.evaluate_run(str(child), split='validation', device=p['resolved_device'])
                write_json(report_path, validation)
                export = child / 'training/inference.pt'
            _save(path, record, validation=validation)
            finish('model_validation'); stage('register')
            if p['model'] == 'resolution-flexible':
                from pdeflow.coordinate.registry import register_checkpoint
                register_checkpoint(export, record['workspace'])
            # Fixed model already belongs to the managed run registry. Registration
            # never changes the user's selected reconstruction model.
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
    for entry in fixed_model_entries(workspace):
        try:
            cp, c, _ = _read_checkpoint(entry['path']); identifier = file_digest(entry['path'])[:32]
            physical = get_specialist(c.specialist).default_problem(c.grid)
            compatibility = compatibility_report(p, c) if p else {'compatible': False, 'summary': 'No current executable problem; use the model training problem.'}
            result[identifier] = {'id': identifier, 'model': 'fixed-grid', 'type': MODELS['fixed-grid'], 'problem': physical.to_dict(),
                                  'updates': {'fixed': cp['progress']}, 'status': entry.get('run_state', 'available export'),
                                  'path': entry['path'], 'compatibility': compatibility, 'available': True}
        except (ValueError, RuntimeError, OSError, KeyError):
            continue
    for entry in coordinate_runs(workspace):
        if entry.get('stage') != 'flow' or not entry.get('available'):
            continue
        c = CoordinateConfig(**entry['config']).validate()
        from pdeflow.equations import heat_problem
        physical = heat_problem(data_config(c).grid, sum(c.diffusivity_range) / 2)
        compatible, reason = False, 'No current executable problem; use the model training problem.'
        if p:
            try:
                c.compatible(p); compatible, reason = True, 'Physical contract matches; query resolution may vary.'
            except ValueError as error:
                reason = str(error)
        identifier = entry['model_id']
        result[identifier] = {'id': identifier, 'model': 'resolution-flexible', 'type': MODELS['resolution-flexible'],
                              'problem': physical.to_dict(), 'updates': {'representation': entry.get('dependency', {}).get('progress', c.ae_steps), 'flow': entry['progress']},
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
    request = from_problem(ProblemSpec.from_dict(entry['problem']), seed=53).request
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
        directory = path / 'validation' / str(case)
        problem = ProblemSpec.from_dict(read_json(directory / 'problem.json'))
        with np.load(directory / 'preview.npz', allow_pickle=False) as data:
            predicted = data['prediction'].reshape(1, len(problem.grid.times), *problem.grid.shape)
            reference = data['reference'].reshape(predicted.shape)
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


def coordinate_stages(config, path, stages='both', mode='new', dependency=None, gate_ratio=1., override_gate=False, stop_after=None, event=None, names=('autoencoder','flow'),stage_hook=None,finish_hook=None,should_stop=None,local_validation=False):
    """Shared two-stage driver for procedural local/CLI/notebook training.

    Uses the same train, representation evaluation and quality-gate semantics as
    execute(), with bounded representative evaluation for large populations.
    """
    from pdeflow.coordinate.contracts import CoordinateConfig
    from pdeflow.coordinate.training import train, read_checkpoint
    from pdeflow.coordinate.validation import representation,flow as evaluate_flow
    from pdeflow.coordinate.optimization import options
    c=CoordinateConfig(**config).validate();path=Path(path);ae=path/names[0];flow=path/names[1];o=options(c)
    def stage(key):
        if stage_hook:stage_hook(key)
    def finish(key):
        if finish_hook:finish_hook(key)
    def run_stage(stage,root,dependency=None):
        resume=(root/'run.json').exists()
        if resume and (root/'resume.pt').exists():
            cp,old=read_checkpoint(root/'resume.pt',stage)
            from pdeflow.coordinate.optimization import resume_contract
            resume_contract(cp,c,stage,mode=='extend')
            total=c.ae_steps if stage=='autoencoder' else c.flow_steps
            if cp['completed'] and cp['progress']==total:return read_json(root/'run.json')
        return train(config,root,stage,dependency,resume,stop_after,should_stop=should_stop or (lambda:(path/'stop-request.json').exists()),event=event,extend=mode=='extend')
    if stages in ('both','representation'):
        if (flow/'resume.pt').exists() and (ae/'resume.pt').exists() and read_checkpoint(ae/'resume.pt')[0]['progress']<c.ae_steps:
            raise ValueError('Extending a representation changes frozen flow targets. Use a new run for a new representation; extend flow alone to preserve the existing target.')
        stage('representation');r=run_stage('autoencoder',ae)
        if r['state']!='completed':return {'state':'interrupted','stage':'autoencoder','progress':r['progress']}
        finish('prepare');finish('representation')
    dependency=dependency or ae/'autoencoder.pt'
    stage('representation_validation')
    if not Path(dependency).exists():raise ValueError('Choose a completed compatible representation run for flow-only training')
    evaluation_path=path/'representation-validation.json'
    if evaluation_path.exists() and read_json(evaluation_path)['checkpoint_sha256']==file_digest(dependency):report=read_json(evaluation_path)
    else:
        if evaluation_path.exists():evaluation_path.rename(path/f'representation-validation-{uuid.uuid4().hex[:8]}.json')
        report=representation(dependency,evaluation_path,c.runtime['device'],subset=min(c.validation_count,max(len(c.distribution.get('families',{})) or 6,o['broad_subset'])))
    quality=representation_quality(dependency,report,gate_ratio);write_json(path/'representation-quality.json',quality)
    finish('representation_validation')
    if stages=='representation':return {'state':'completed','stage':'representation_validation','quality':quality}
    if not quality['passed'] and not override_gate:return {'state':'needs_review','stage':'representation_gate','quality':quality,'representation_validation':report,'meaning':'Saved representation did not beat the constant baseline. Flow not started. Explicit recorded override required.'}
    stage('flow');r=run_stage('flow',flow,dependency)
    if r['state']!='completed':return {'state':'interrupted','stage':'flow','progress':r['progress'],'quality':quality}
    finish('flow');stage('model_validation')
    if local_validation:
        evaluation=_coordinate_validation(path,config,c.runtime['device']);write_json(path/'model-validation.json',evaluation)
        return {'state':'completed','quality':quality,'representation_validation':report,'gate_override':override_gate,'validation':evaluation,'export':str(flow/'coordinate.pt')}
    previous=path/'flow-validation.json'
    if previous.exists() and read_json(previous).get('checkpoint_sha256')==file_digest(flow/'coordinate.pt'):evaluation=read_json(previous)
    else:
        if previous.exists():
            archive=path/'evaluations'/uuid.uuid4().hex[:12];archive.mkdir(parents=True)
            previous.rename(archive/'flow-validation.json')
            if previous.with_suffix('.npz').exists():previous.with_suffix('.npz').rename(archive/'flow-validation.npz')
        evaluation=evaluate_flow(flow/'coordinate.pt',previous,c.runtime['device'],subset=min(c.validation_count,6),steps=2 if c.flow_steps<=4 else 8)
    return {'state':'completed','stage':'evaluated','quality':quality,'representation_validation':report,'gate_override':override_gate,'validation':evaluation,'export':str(flow/'coordinate.pt')}


if __name__ == '__main__':
    execute(Path(sys.argv[1]))
