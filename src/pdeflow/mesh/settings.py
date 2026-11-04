"""Explicit new-run optimizer/batch settings, shared by UI, Colab and trainer."""
from dataclasses import replace
import math


def resolved(c, stage):
    overrides = c.stage_settings.get(stage, {})
    batch = overrides.get('batch_size', c.batch_size)
    micro = overrides.get('microbatch', c.microbatch)
    if any(type(value) is not int or value<1 for value in (batch,micro)):
        raise ValueError(stage+': positive integer effective batch and microbatch required')
    optimizer = {**c.optimizer, **overrides.get('optimizer', {})}
    return {'learning_rate': overrides.get('learning_rate', c.learning_rate),
            'effective_batch': batch, 'microbatch': micro,
            'accumulation': math.ceil(batch / micro),
            'last_microbatch': batch - (math.ceil(batch / micro) - 1) * micro,
            'optimizer': {'name': 'AdamW', 'weight_decay': 0., **optimizer},
            'schedule_horizon': c.budgets[stage], 'successful_update_budget': c.budgets[stage]}


def validate(c):
    if set(c.stage_settings) - set(c.budgets):
        raise ValueError('Unknown mesh stage override')
    if c.batch_policy not in ('keep_updates', 'keep_examples'):
        raise ValueError('Choose explicit keep_updates or keep_examples batch policy')
    allowed = {'schedule', 'warmup', 'minimum_ratio', 'clip', 'max_consecutive_overflows', 'initial_scale'}
    for stage, values in c.stage_settings.items():
        if not isinstance(values, dict) or set(values) - {'learning_rate', 'batch_size', 'microbatch', 'optimizer'}:
            raise ValueError('Stage overrides support learning_rate, batch_size, microbatch and optimizer only')
        if 'optimizer' in values and (not isinstance(values['optimizer'],dict) or set(values['optimizer'])-allowed):
            raise ValueError('Invalid per-stage optimizer setting; AdamW and weight decay are fixed')
    for stage in c.kind['stages']:
        r = resolved(c, stage); o = r['optimizer']
        if any(type(r[k]) is not int or r[k] < 1 for k in ('effective_batch', 'microbatch')) or r['microbatch'] > r['effective_batch']:
            raise ValueError(stage + ': positive microbatch <= effective batch required')
        if type(r['learning_rate']) not in (float,int) or not math.isfinite(r['learning_rate']) or r['learning_rate'] <= 0:
            raise ValueError(stage + ': positive finite learning rate required')
        if set(o) - allowed - {'name', 'weight_decay'} or o.get('schedule', 'constant') not in ('constant', 'cosine'):
            raise ValueError(stage + ': invalid optimizer settings')
        warmup = o.get('warmup', 0)
        if type(warmup) is not int or not 0 <= warmup < c.budgets[stage]:
            raise ValueError(stage + ': warmup must be below the resolved successful-update budget; change it explicitly')
        if not 0 <= o.get('minimum_ratio', .1) <= 1 or not math.isfinite(o.get('clip', 10.)) or o.get('clip', 10.) <= 0:
            raise ValueError(stage + ': invalid minimum LR ratio or clipping threshold')
        if type(o.get('max_consecutive_overflows', 8)) is not int or o.get('max_consecutive_overflows', 8) < 1 or not math.isfinite(o.get('initial_scale', 65536.)) or o.get('initial_scale', 65536.) <= 0:
            raise ValueError(stage + ': invalid scaler recovery settings')


def configure(c, *, learning_rate=None, effective_batch=None, microbatch=None,
              stages=None, budget_policy='keep_updates'):
    """NEW experiments only. Never scale LR or warmup implicitly.

    keep_examples rounds each stage's original example budget UP to whole updates.
    Caller must pass the original preset/config, not repeatedly apply this conversion.
    """
    if budget_policy not in ('keep_updates', 'keep_examples'):
        raise ValueError('Choose keep_updates or keep_examples')
    changes = {'stage_settings': c.stage_settings if stages is None else stages, 'batch_policy': budget_policy}
    for name, value in (('learning_rate', learning_rate), ('batch_size', effective_batch), ('microbatch', microbatch)):
        if value is not None: changes[name] = value
    new = replace(c, **changes)
    budgets = dict(c.budgets); reference = {}
    for stage in c.kind['stages']:
        before, after = resolved(c, stage), resolved(new, stage)
        if type(after['effective_batch']) is not int or after['effective_batch'] < 1:
            raise ValueError('Positive integer effective batch required')
        examples = before['effective_batch'] * c.budgets[stage]
        if budget_policy == 'keep_examples': budgets[stage] = math.ceil(examples / after['effective_batch'])
        reference[stage] = {'original_batch': before['effective_batch'], 'original_updates': c.budgets[stage],
                            'original_examples': examples, 'resolved_examples': after['effective_batch'] * budgets[stage],
                            'rounding_extra_examples': after['effective_batch'] * budgets[stage] - examples if budget_policy == 'keep_examples' else 0}
    return replace(new, budgets=budgets, batch_reference=reference).validate()


def summary(c):
    return {'policy': c.batch_policy, 'reference': c.batch_reference,
            'stages': {s: resolved(c, s) for s in c.kind['stages']},
            'semantics': 'Per-example loss sum / effective batch, including a shorter final microbatch. No implicit learning-rate or warmup scaling. Budgets count successful optimizer updates.'}
