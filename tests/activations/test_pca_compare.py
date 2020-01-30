import os
import numpy as np
from pathlib import Path

import pytest

from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation import ModelCommitment
from tests.activations.test___init__ import pytorch_alexnet, keras_vgg19


@pytest.mark.parametrize(["model_ctr", "layers","image_name", "pca_components"], [
    # pytest.param(pytorch_custom, ['linear', 'relu2']),
   (pytorch_alexnet, ['features.12', 'classifier.5'], 'rgb.jpg',1000),
   (keras_vgg19, ['block3_pool'], 'rgb.jpg',1000),
])
def test_compare_PCA_activations(model_ctr, layers, image_name, pca_components):
    stimuli_paths = [os.path.join(os.path.dirname(__file__), image_name)]

    activations_extractor = model_ctr()
    if pca_components:
        LayerPCA.hook(activations_extractor, pca_components, True)
    activations_batched = activations_extractor.from_paths(stimuli_paths=stimuli_paths,
                                                           layers=layers)

    activations_extractor = model_ctr()
    if pca_components:
        LayerPCA.hook(activations_extractor, pca_components, False)
    activations = activations_extractor.from_paths(stimuli_paths=stimuli_paths,
                                                   layers=layers)
    assert activations is not None and activations_batched is not None
    assert len(np.unique(activations['layer'])) == len(np.unique(activations_batched['layer']))
    import gc
    gc.collect()
    return activations

@pytest.mark.parametrize(["model_ctr", "layers"], [
    # pytest.param(pytorch_custom, ['linear', 'relu2']),
   (pytorch_alexnet, ['features.12', 'classifier.5']),
   (keras_vgg19, ['block3_pool']),
])
def test_compare_PCA_commitments(model_ctr, layers):
    activations_model = model_ctr()

    for region in ('V1', 'V2', 'V4', 'IT'):
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=layers, batch_pca=True)
        brain_model.commit_region(region)
        brain_model.start_recording(region, [(70, 170)])
        predictions_batched = brain_model.look_at([Path(__file__).parent / 'rgb.jpg'])

        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=layers, batch_pca=False)
        brain_model.commit_region(region)
        brain_model.start_recording(region, [(70, 170)])
        predictions = brain_model.look_at([Path(__file__).parent / 'rgb.jpg'])

        assert set(predictions['region'].values) == {predictions_batched['region'].values}
        assert set(predictions['layer'].values) == {predictions_batched['layer'].values}
