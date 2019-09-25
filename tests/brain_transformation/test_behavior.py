import os

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from brainio_base.stimuli import StimulusSet
from brainscore.benchmarks.behavioral import DicarloRajalingham2018I2n
from brainscore.model_interface import BrainModel
from model_tools.brain_transformation import ModelCommitment, ProbabilitiesMapping
from ..models import pytorch_custom


class TestLogitsBehavior:
    @pytest.mark.parametrize(['model_ctr'], [(pytorch_custom,)])
    def test_creates_synset(self, model_ctr):
        np.random.seed(0)
        activations_model = model_ctr()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=None, behavioral_readout_layer='dummy')  # not needed
        stimuli = StimulusSet({'image_id': ['abc123']})
        stimuli.image_paths = {'abc123': os.path.join(os.path.dirname(__file__), 'rgb1.jpg')}
        stimuli.name = 'test_logits_behavior.creates_synset'
        brain_model.start_task(BrainModel.Task.label, 'imagenet')
        synsets = brain_model.look_at(stimuli)
        assert len(synsets) == 1
        assert synsets[0].startswith('n')


class TestProbabilitiesMapping:
    def test_creates_probabilities(self):
        activations_model = pytorch_custom()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=None, behavioral_readout_layer='relu2')
        fitting_stimuli = StimulusSet({'image_id': ['rgb1', 'rgb2'], 'image_label': ['label1', 'label2']})
        fitting_stimuli.image_paths = {'rgb1': os.path.join(os.path.dirname(__file__), 'rgb1.jpg'),
                                       'rgb2': os.path.join(os.path.dirname(__file__), 'rgb2.jpg')}
        fitting_stimuli.name = 'test_probabilities_mapping.creates_probabilities'
        brain_model.start_task(BrainModel.Task.probabilities, fitting_stimuli)
        probabilities = brain_model.look_at(fitting_stimuli)
        np.testing.assert_array_equal(probabilities.dims, ['presentation', 'choice'])
        np.testing.assert_array_equal(probabilities.shape, [2, 2])
        np.testing.assert_array_almost_equal(probabilities.sel(image_id='rgb1', choice='label1').values,
                                             probabilities.sel(image_id='rgb2', choice='label2').values)
        assert probabilities.sel(image_id='rgb1', choice='label1') + \
               probabilities.sel(image_id='rgb1', choice='label2') == approx(1)


@pytest.mark.private_access
class TestI2N:
    @pytest.mark.parametrize(['model', 'expected_score'],
                             [
                                 ('alexnet', .253),
                                 ('resnet34', .37787),
                                 ('resnet18', .3638),
                             ])
    def test_model(self, model, expected_score):
        class UnceiledBenchmark(DicarloRajalingham2018I2n):
            def __call__(self, candidate: BrainModel):
                candidate.start_task(BrainModel.Task.probabilities, self._fitting_stimuli)
                probabilities = candidate.look_at(self._assembly.stimulus_set)
                score = self._metric(probabilities, self._assembly)
                return score

        benchmark = UnceiledBenchmark()
        # features
        feature_responses = pd.read_pickle(
            os.path.join(os.path.dirname(__file__), f'identifier={model},stimuli_identifier=objectome-240.pkl'))['data']
        feature_responses['image_id'] = 'stimulus_path', [os.path.splitext(os.path.basename(path))[0]
                                                          for path in feature_responses['stimulus_path'].values]
        feature_responses = feature_responses.stack(presentation=['stimulus_path'])
        assert len(np.unique(feature_responses['layer'])) == 1  # only penultimate layer

        class PrecomputedFeatures:
            def __init__(self, precomputed_features):
                self.features = precomputed_features

            def __call__(self, stimuli, layers):
                np.testing.assert_array_equal(layers, ['behavioral-layer'])
                self_image_ids = self.features['image_id'].values.tolist()
                indices = [self_image_ids.index(image_id) for image_id in stimuli['image_id'].values]
                features = self.features[{'presentation': indices}]
                return features

        # evaluate candidate
        transformation = ProbabilitiesMapping(identifier=f'TestI2N.{model}',
                                              activations_model=PrecomputedFeatures(feature_responses),
                                              layer='behavioral-layer')
        score = benchmark(transformation)
        score = score.sel(aggregation='center')
        assert score == approx(expected_score, abs=0.005), f"expected {expected_score}, but got {score}"
