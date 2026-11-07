"""Small, pure summaries of recorded mesh metrics; no model, RNG or training calls."""
import math


def successful(row):
    value=row.get('training_loss')
    return (isinstance(value,(int,float)) and math.isfinite(value)
            and not row.get('optimizer_update_skipped',False)
            and not row.get('failed_attempt',False) and row.get('gradients_finite',True))


def success_rows(rows):
    # Reconnects may repeat the same saved row. Never count it twice.
    unique={}
    for r in rows:
        if successful(r):unique[r['step']]=r
    return [unique[k] for k in sorted(unique)]


def training_summary(rows,window=100):
    selected=success_rows(rows)[-window:]
    families={}
    for family in ('heat','wave'):
        parts=[r['family_objective'][family] for r in selected if family in r.get('family_objective',{})]
        total=sum(p['sum'] for p in parts);count=sum(p['examples'] for p in parts)
        families[family]={'sum':total,'examples':count,'mean':total/count if count else None,
                          'recorded_updates':len(parts),'missing_updates':len(selected)-len(parts)}
    return {'window':window,'count':len(selected),'mean':sum(r['training_loss'] for r in selected)/len(selected) if selected else None,
            'first_update':selected[0]['step'] if selected else None,'last_update':selected[-1]['step'] if selected else None,
            'label':f'Training objective: rolling mean, last {window} successful updates',
            'family_objective':families,'quantity_objective':'No general per-quantity split of the optimized objective; use decoded-field validation. Existing components retain their original definitions.'}


def rolling_series(rows,window=100):
    from collections import deque
    active=deque();total=0.;series=[]
    for row in success_rows(rows):
        active.append(row['training_loss']);total+=row['training_loss']
        if len(active)>window:total-=active.popleft()
        series.append({'step':row['step'],'mean':total/len(active),'count':len(active)})
    return series


def family_contributions(values,families,totals=None):
    """Actual detached per-example objective sums, not means of microbatch means."""
    totals=totals if totals is not None else {f:{'sum':0.,'examples':0} for f in ('heat','wave')}
    for value,family in zip(values,families,strict=True):
        if not math.isfinite(value):raise ValueError('Nonfinite family objective contribution')
        item=totals[('heat','wave')[int(family)]];item['sum']+=value;item['examples']+=1
    return totals


def attach(record,validation_history=None):
    """Read-only extension; old rows never acquire invented family measurements."""
    record=dict(record);window=record.get('training_smoothing_window',100)
    source=record.get('smoothing_history',record.get('history',[]))
    record['training_summary']=training_summary(source,window)
    record['training_series']=rolling_series(source,window)[-200:]
    if validation_history is None:validation_history=record.get('validation_history',[])
    record['validation_history']=validation_history
    valid=[r for r in validation_history if r.get('stage',record.get('stage'))==record.get('stage')]
    record['validation_history']=valid
    record['latest_validation']=valid[-1] if valid else None
    comparable=[r for r in valid if isinstance(r.get('uniform_domain_normalized_mse'),(int,float)) and math.isfinite(r['uniform_domain_normalized_mse'])]
    record['best_validation']=min(comparable,key=lambda r:r['uniform_domain_normalized_mse']) if comparable else None
    interval=record.get('validation_every')
    step=record.get('successful_updates',record.get('progress',0)) or 0
    record['next_validation_update']=((step//interval)+1)*interval if interval else None
    if interval and record.get('budget') is not None and step<record['budget']:
        record['next_validation_update']=min(record['next_validation_update'],record['budget'])
    if record.get('budget') is not None and step>=record['budget']:record['next_validation_update']=None
    return record


def compact(row):
    return {k:row[k] for k in ('step','training_loss','family_objective') if k in row}


def validation_summary(result):
    keys=('stage','update','architecture','run_id','validation_id','model_state_sha256','validation_spec_sha256',
          'validation_protocol','uniform_domain_normalized_mse','balanced_family_normalized_mse',
          'decoded_field_mse','balanced_family_physical_mse','groups','case_count','unique_case_count',
          'evaluated_points','generated_samples','state_predictions','reason','prediction_scope')
    return {k:result[k] for k in keys if k in result}


def legacy_validation(rows,stage):
    return [{'stage':stage,'update':r['step'],'uniform_domain_normalized_mse':r['validation_loss'],'groups':{},'legacy':True}
            for r in rows if isinstance(r.get('validation_loss'),(int,float)) and math.isfinite(r['validation_loss'])]


def recorded_validation(rows,stage,last=None):
    known=legacy_validation(rows,stage)
    if known:return known
    if last and isinstance(last.get('uniform_domain_normalized_mse'),(int,float)):
        # Keep the scalar even when old records never attached an update to it.
        return [validation_summary({**last,'stage':stage})]
    return []
