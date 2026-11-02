"""Actual effective-batch gradients/objectives must agree with logged B components."""
from dataclasses import replace
import pytest
import torch
from pdeflow.colab import training_preset
from pdeflow.coordinate.training import train,read_checkpoint

@pytest.mark.parametrize('beta',[0.,.01])
def test_full_batch_flow_and_weighted_endpoint_components(tmp_path,beta):
    torch.set_num_threads(2)
    c=training_preset('pipeline-check','cpu','fp32',{'batch_size':5,'encoder_points':16,'query_points':16,'training_options':{'microbatch':3,'accumulation':2},'joint':{'observation_auxiliary':{'beta':beta,'minimum_s':0.,'sensor_budget':4}}})
    a=train(c.to_dict(),tmp_path/'a1','autoencoder');e=train(c.to_dict(),tmp_path/'a2','evolution',a['export'])
    for execution in ('batched','accumulated'):
        config=replace(c,training_options={**c.training_options,'execution':execution})
        result=train(config.to_dict(),tmp_path/execution,'flow',e['export']);cp,_=read_checkpoint(result['export'])
        for row in cp['history']:
            components=row['components']
            assert row['training_loss']==pytest.approx(components['flow_matching']+components['objective_observation_auxiliary'],rel=3e-6,abs=1e-6)
            if beta==0:assert components['objective_observation_auxiliary']==0
            else:
                assert components['observation_population']==components['observation_values']
                assert components['objective_observation_auxiliary']==pytest.approx(beta*components['observation_endpoint_loss'],rel=3e-6,abs=1e-6)
