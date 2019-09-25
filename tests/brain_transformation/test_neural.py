import os

import pytest

from brainio_base.stimuli import StimulusSet
from brainscore.assemblies.public import load_assembly
from model_tools.brain_transformation import ModelCommitment, LayerMappedModel
from ..models import pytorch_custom


class TestLayerSelection:
    @pytest.mark.parametrize(['model_ctr', 'layers', 'expected_layer', 'assembly_identifier', 'region'],
                             [(pytorch_custom, ['linear', 'relu2'], 'relu2', 'dicarlo.Majaj2015.lowvar.IT', 'IT')])
    def test_commit_record(self, model_ctr, layers, expected_layer, assembly_identifier, region):
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=layers)
        assembly = load_assembly(assembly_identifier, average_repetition=False)
        brain_model.commit_region(region, assembly, assembly_stratification='category_name')

        brain_model.start_recording(region, [(70, 170)])
        predictions = brain_model.look_at(assembly.stimulus_set)
        assert set(predictions['region'].values) == {region}
        assert set(predictions['layer'].values) == {expected_layer}


class TestLayerMappedModel:
    @pytest.mark.parametrize(['model_ctr', 'layers', 'region'], [
        (pytorch_custom, 'relu2', 'IT'),
        (pytorch_custom, ['linear', 'relu2'], 'IT'),
    ])
    def test_commit(self, model_ctr, layers, region):
        activations_model = model_ctr()
        layer_model = LayerMappedModel(identifier=activations_model.identifier, activations_model=activations_model)
        layer_model.commit(region, layers)

        layer_model.start_recording(region)
        stimulus_set = StimulusSet([{'image_id': 'test'}])
        stimulus_set.image_paths = {'test': os.path.join(os.path.dirname(__file__), 'rgb1.jpg')}
        stimulus_set.name = self.__class__.__name__
        predictions = layer_model.look_at(stimulus_set)
        assert set(predictions['region'].values) == {region}
        assert set(predictions['layer'].values) == {layers} if isinstance(layers, str) else set(layers)
