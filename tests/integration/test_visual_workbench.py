"""Visual payloads use real saved inputs/results, with bounded projections."""
import copy
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pdeflow.runtime import read_json, write_json
from pdeflow.workbench.api import create_app
from pdeflow.workbench.visuals import bounded_rows, input_preview, result_plots, compare_results, training_runs

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def managed(tmp_path_factory):
    from pdeflow.local import start_training
    root=tmp_path_factory.mktemp("visual-managed")
    start_training(preset="quickstart-heat-2d",workspace=root,device="cpu",steps=2,name="visual-test")
    return root


def test_actual_darcy_input_slices_and_unknown_regions():
    from pdeflow.runtime import Config
    from pdeflow.specialists import get_specialist
    c=Config.load("configs/experiments/darcy_cpu_3d.json")
    p=get_specialist("darcy").default_problem(c.grid).to_dict()
    coords=c.grid.coordinates()
    x=np.meshgrid(*[axis.numpy() for axis in coords],indexing="ij")
    p["known_fields"]={"f":(np.sin(2*np.pi*x[0])+np.cos(2*np.pi*x[1])+np.sin(2*np.pi*x[2])).tolist()}
    rows=[]
    from pdeflow.core import ProblemSpec
    from pdeflow.observations import ObservationSet
    from pdeflow.workbench.observations import observation_rows
    problem=ProblemSpec.from_dict(p)
    rows=[{"field":"a",**{name:float(problem.grid.coordinates()[i][3]) for i,name in enumerate(["x","y","z"])},"value":1.0}]
    for axis in range(3):
        preview=input_preview(p,rows,'f',0,axis,3)
        np.testing.assert_array_equal(preview['values'],np.take(np.asarray(p['known_fields']['f']),3,axis=axis))
        assert preview['physical_time'] is None
        assert preview['axes']==[a for i,a in enumerate(['x','y','z']) if i!=axis]
        assert preview['slice_coordinate']==float(problem.grid.coordinates()[axis][3])
    unknown=input_preview(p,rows,'a')
    assert unknown['values'] is None and 'unknown target' in unknown['role']
    bad=copy.deepcopy(rows);bad[0]['x']=.123456
    invalid=input_preview(p,bad)
    assert not invalid['valid'] and invalid['issues'][0]['row']==0
    assert not invalid['points']
    with pytest.raises(ValueError):input_preview(p,rows,'reference')


def test_bounded_history_preserves_extrema_and_recorded_checkpoints(managed):
    rows=[{'step':i,'training_loss':float(i%19),'validation_loss':float(i%17)} for i in range(10000)]
    bounded,changed=bounded_rows(rows)
    assert changed and len(bounded)<=602
    assert bounded[0]==rows[0] and bounded[-1]==rows[-1]
    assert max(r['training_loss'] for r in bounded)==18
    histories=training_runs(managed)
    assert histories
    for record in histories:
        assert len(record['history'])<=602
        assert set(record['checkpoint_positions']).issubset({0,*[row['step'] for row in record['history']]})
        assert 'not reconstruction error' in record['meaning']


def test_real_job_visual_routes_weighted_predictions_and_reference_isolation(tmp_path,managed):
    from test_workbench import wait
    workspace=tmp_path/'workspace'
    app=create_app(model_roots=[managed/'runs'],output_root=tmp_path/'jobs',data_root=tmp_path/'data',workspace=workspace)
    with TestClient(app) as client:
        model=client.get('/api/models').json()[0]
        demo=client.post('/api/demos',json={'model_id':model['id'],'sensors':4}).json()
        preview=client.post('/api/input-preview',json={'problem':demo['problem'],'rows':demo['rows']})
        assert preview.status_code==200 and preview.json()['values'] is None
        assert client.get('/api/training-runs').json()==[]
        assert {'heat-2d-medium','heat-2d-large','heat-2d-1024-cuda','heat-2d-native-cpu'}.issubset({p['name'] for p in client.get('/api/training-presets').json()})
        request={'model_id':model['id'],'problem':demo['problem'],'rows':demo['rows'],'mode':'importance','inference':{'steps':3,'particles':3,'fk':{}},'submission_token':'visual'}
        job=client.post('/api/jobs',json=request)
        assert job.status_code==202,job.text
        identifier=job.json()['id']
        assert wait(client,identifier)['status']=='succeeded'
        plots=client.get(f'/api/jobs/{identifier}/plots').json()
        assert plots['weighted'] and plots['reference_used'] is False
        saved=client.get(f'/api/jobs/{identifier}/input-preview').json()
        assert saved['values'] is None and saved['total_observations']==len(demo['rows'])
        with np.load(io.BytesIO(client.get(f'/api/jobs/{identifier}/artifacts/samples.npz').content)) as arrays:
            from pdeflow.core import ProblemSpec
            from pdeflow.observations import ObservationSet
            import torch
            directory=tmp_path/'jobs'/identifier/'result'
            p=ProblemSpec.from_dict(read_json(directory/'problem.json'))
            obs=ObservationSet.from_dict(read_json(directory/'observations.json'),p.grid)
            predictions=obs.operator.apply(torch.from_numpy(arrays['samples'][0])).numpy()
            expected=(predictions*arrays['weights'][0,:,None]).sum(0)
            np.testing.assert_allclose([pt['predicted'] for pt in plots['points']],expected,rtol=1e-6)
        assert client.post(f'/api/jobs/{identifier}/reference',json={'reference_id':demo['reference_id']}).status_code==200
        assert client.get(f'/api/jobs/{identifier}/plots').json()==plots
        assert client.post(f'/api/jobs/{identifier}/reference',json={'reference_id':None}).status_code==200
        assert client.get(f'/api/jobs/{identifier}/plots').json()==plots
        first=tmp_path/'jobs'/identifier/'result'
        import shutil
        other=tmp_path/'other';shutil.copytree(first,other)
        compare=compare_results([('first',first),('other',other)])
        assert compare['compatible'] and not compare['same_reference']
        observations=read_json(other/'observations.json');observations['noise_std']=.9
        write_json(other/'observations.json',observations)
        with pytest.raises(ValueError,match='different equations'):compare_results([('first',first),('other',other)])
        assert client.post('/api/compare',json={'ids':[identifier,identifier]}).status_code==422
