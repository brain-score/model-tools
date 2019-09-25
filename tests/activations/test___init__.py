import functools
import os
import pickle

import numpy as np
import pytest

from brainio_base.stimuli import StimulusSet
from model_tools.activations import PytorchWrapper
from model_tools.activations.core import flatten
from model_tools.activations.pca import LayerPCA
from ..models import \
    pytorch_custom, pytorch_alexnet, pytorch_alexnet_resize, \
    keras_vgg19, \
    tfslim_custom, tfslim_vgg16


def unique_preserved_order(a):
    _, idx = np.unique(a, return_index=True)
    return a[np.sort(idx)]


class TestActivations:
    @pytest.mark.parametrize("image_name", ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png'])
    @pytest.mark.parametrize(["pca_components", "logits"], [(None, True), (None, False), (5, False)])
    @pytest.mark.parametrize(["model_ctr", "layers"], [
        pytest.param(pytorch_custom, ['linear', 'relu2']),
        pytest.param(pytorch_alexnet, ['features.12', 'classifier.5'], marks=pytest.mark.memory_intense),
        pytest.param(keras_vgg19, ['block3_pool'], marks=pytest.mark.memory_intense),
        pytest.param(tfslim_custom, ['my_model/pool2'], marks=pytest.mark.memory_intense),
        pytest.param(tfslim_vgg16, ['vgg_16/pool5'], marks=pytest.mark.memory_intense),
    ])
    def test_from_image_path(self, model_ctr, layers, image_name, pca_components, logits):
        stimuli_paths = [os.path.join(os.path.dirname(__file__), image_name)]

        activations_extractor = model_ctr()
        if pca_components:
            LayerPCA.hook(activations_extractor, pca_components)
        activations = activations_extractor.from_paths(stimuli_paths=stimuli_paths,
                                                       layers=layers if not logits else None)

        assert activations is not None
        assert len(activations['stimulus_path']) == 1
        assert len(np.unique(activations['layer'])) == len(layers) if not logits else 1
        if logits and not pca_components:
            assert len(activations['neuroid']) == 1000
        elif pca_components is not None:
            assert len(activations['neuroid']) == pca_components * len(layers)
        import gc
        gc.collect()
        return activations

    @pytest.mark.parametrize("pca_components", [None, 5])
    @pytest.mark.parametrize(["model_ctr", "layers"], [
        pytest.param(pytorch_custom, ['linear', 'relu2']),
        pytest.param(pytorch_alexnet, ['features.12', 'classifier.5'], marks=pytest.mark.memory_intense),
        pytest.param(keras_vgg19, ['block3_pool'], marks=pytest.mark.memory_intense),
        pytest.param(tfslim_custom, ['my_model/pool2'], marks=pytest.mark.memory_intense),
        pytest.param(tfslim_vgg16, ['vgg_16/pool5'], marks=pytest.mark.memory_intense),
    ])
    def test_from_stimulus_set(self, model_ctr, layers, pca_components):
        image_names = ['rgb.jpg', 'grayscale.png', 'grayscale2.jpg', 'grayscale_alpha.png']
        stimulus_set = StimulusSet([{'image_id': image_name, 'some_meta': image_name[::-1]}
                                    for image_name in image_names])
        stimulus_set.image_paths = {image_name: os.path.join(os.path.dirname(__file__), image_name)
                                    for image_name in image_names}

        activations_extractor = model_ctr()
        if pca_components:
            LayerPCA.hook(activations_extractor, pca_components)
        activations = activations_extractor.from_stimulus_set(stimulus_set, layers=layers, stimuli_identifier=False)

        assert activations is not None
        assert set(activations['image_id'].values) == set(image_names)
        assert all(activations['some_meta'].values == [image_name[::-1] for image_name in image_names])
        assert len(np.unique(activations['layer'])) == len(layers)
        if pca_components is not None:
            assert len(activations['neuroid']) == pca_components * len(layers)

    @pytest.mark.memory_intense
    @pytest.mark.parametrize("pca_components", [None, 1000])
    def test_exact_activations(self, pca_components):
        activations = self.test_from_image_path(
            model_ctr=pytorch_alexnet_resize, layers=['features.12', 'classifier.5'],
            image_name='rgb.jpg', pca_components=pca_components, logits=False)
        with open(os.path.join(os.path.dirname(__file__), f'alexnet-rgb-{pca_components}.pkl'), 'rb') as f:
            target = pickle.load(f)['activations']
        assert (activations == target).all()

    @pytest.mark.memory_intense
    @pytest.mark.parametrize(["model_ctr", "internal_layers"], [
        (pytorch_alexnet, ['features.12', 'classifier.5']),
        (keras_vgg19, ['block3_pool']),
        (tfslim_vgg16, ['vgg_16/pool5']),
    ])
    def test_mixed_layer_logits(self, model_ctr, internal_layers):
        stimuli_paths = [os.path.join(os.path.dirname(__file__), 'rgb.jpg')]

        activations_extractor = model_ctr()
        layers = internal_layers + ['logits']
        activations = activations_extractor(stimuli=stimuli_paths, layers=layers)
        assert len(np.unique(activations['layer'])) == len(internal_layers) + 1
        assert set(activations['layer'].values) == set(layers)
        assert unique_preserved_order(activations['layer'])[-1] == 'logits'


@pytest.mark.memory_intense
@pytest.mark.parametrize(["model_ctr", "expected_identifier"], [
    (pytorch_custom, 'MyModel'),
    (pytorch_alexnet, 'AlexNet'),
    (keras_vgg19, 'vgg19'),
])
def test_infer_identifier(model_ctr, expected_identifier):
    model = model_ctr()
    assert model.identifier == expected_identifier


class TestChannels:
    def test_convolution_meta(self):
        model = pytorch_custom()
        activations = model(stimuli=[os.path.join(os.path.dirname(__file__), 'rgb.jpg')], layers=['conv1'])
        assert hasattr(activations, 'channel')
        assert hasattr(activations, 'channel_x')
        assert hasattr(activations, 'channel_y')
        assert len(set(activations['neuroid_id'].values)) == len(activations['neuroid'])

    def test_conv_and_fc(self):
        model = pytorch_custom()
        activations = model(stimuli=[os.path.join(os.path.dirname(__file__), 'rgb.jpg')], layers=['conv1', 'linear'])
        assert set(activations['layer'].values) == {'conv1', 'linear'}


@pytest.mark.timeout(300)
def test_merge_large_layers():
    import torch
    from torch import nn
    from model_tools.activations.pytorch import load_preprocess_images

    class LargeModel(nn.Module):
        def __init__(self):
            super(LargeModel, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv(x)
            x = self.relu(x)
            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    model = PytorchWrapper(model=LargeModel(), preprocessing=preprocessing)
    activations = model(stimuli=[os.path.join(os.path.dirname(__file__), 'rgb.jpg')] * 64, layers=['conv', 'relu'])
    assert len(activations['neuroid']) == 394272
    assert len(set(activations['neuroid_id'].values)) == len(activations['neuroid'])
    assert set(activations['layer'].values) == {'conv', 'relu'}


class TestFlatten:
    def test_flattened_shape(self):
        A = np.random.rand(2560, 256, 6, 6)
        flattened = flatten(A)
        assert np.prod(flattened.shape) == np.prod(A.shape)
        assert flattened.shape[0] == A.shape[0]
        assert len(flattened.shape) == 2

    def test_indices_shape(self):
        A = np.random.rand(2560, 256, 6, 6)
        _, indices = flatten(A, return_index=True)
        assert len(indices.shape) == 2
        assert indices.shape[0] == np.prod(A.shape[1:])
        assert indices.shape[1] == 3  # for 256, 6, 6

    def test_match_flatten(self):
        A = np.random.rand(10, 256, 6, 6)
        flattened, indices = flatten(A, return_index=True)
        for layer in range(A.shape[0]):
            for i in range(np.prod(A.shape[1:])):
                value = flattened[layer][i]
                index = indices[i]
                assert A[layer][tuple(index)] == value

    def test_inverse(self):
        A = np.random.rand(2560, 256, 6, 6)
        flattened = flatten(A)
        A_ = np.reshape(flattened, [flattened.shape[0], 256, 6, 6])
        assert A.shape == A_.shape
        assert (A == A_).all()
