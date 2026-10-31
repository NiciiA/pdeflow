import numpy as np
import pytest
from fastapi.testclient import TestClient
from pdeflow.workbench.api import create_app
from pdeflow.training_workflow import plan
from pdeflow.field_losses import settings
from pdeflow.coordinate.dataset import Dataset
from pdeflow.coordinate.contracts import CoordinateConfig


def test_new_plan_default_and_bounded_training_preview(tmp_path):
    request={'problem':'heat-2d','model':'resolution-flexible','budget':'pipeline-check','device':'cpu'}
    result=plan(request);assert result['config']['field_loss']['type']=='region_balanced_mse' and result['config']['field_loss']['alpha']==.5
    with TestClient(create_app(workspace=tmp_path)) as client:
        response=client.post('/api/training/loss-preview',json=request);assert response.status_code==200,response.text
        preview=response.json();assert preview['split']=='train' and len(preview['examples'])==12
        dataset=Dataset(CoordinateConfig(**result['config']).validate(),tmp_path/'preview-calibration')
        assert preview['examples'][0]['training_calibration_scale']>0
        assert 'Preview-only scale estimate' in preview['meaning']
        assert all(e['id'].startswith('train:') for e in preview['examples'])


def test_metrics_match_map_means_and_no_inference(tmp_path):
    with TestClient(create_app(workspace=tmp_path)) as client:
        ref=client.post('/api/playground/reference',json={'preset':'two'}).json()
        b=np.asarray(ref['frames'][0]);prediction=b+.2
        payload={'reference_id':ref['id'],'time':0.,'prediction':prediction.tolist(),'field_loss':{'type':'region_balanced_mse'},'normalization_scale':.3}
        result=client.post('/api/playground/error-metrics',json=payload);assert result.status_code==200,result.text
        r=result.json();assert r['mse']==pytest.approx(.04) and r['rmse']==pytest.approx(.2)
        assert np.mean(r['region_contribution'])==pytest.approx(r['region_balanced_mse'])
        assert np.mean(r['weighted_contribution'])==pytest.approx(r['weighted_mse'])
        assert r['region_count']>=2
        assert client.post('/api/playground/error-metrics',json={**payload,'prediction':[[0],[0]]}).status_code==422
        assert not list((tmp_path/'workbench/jobs').glob('*/request.json'))
