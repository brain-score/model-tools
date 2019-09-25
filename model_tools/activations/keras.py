from collections import OrderedDict
from pathlib import Path

import numpy as np
from xarray import DataArray

from model_tools.activations.core import ActivationsExtractorHelper, collapse_weights, merge_weight_assemblies


class KerasWrapper:
    def __init__(self, model, preprocessing, identifier=None, *args, **kwargs):
        """
        :param model: a keras model with a function `preprocess_input`
            that will later be called on the loaded numpy image
        """
        self._model = model
        identifier = identifier or model.name
        self._extractor = ActivationsExtractorHelper(
            identifier=identifier, get_activations=self.get_activations, preprocessing=preprocessing,
            *args, **kwargs)
        self._extractor.insert_attrs(self)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def get_activations(self, images, layer_names):
        from keras import backend as K
        input_tensor = self._model.input
        layers = [layer for layer in self._model.layers if layer.name in layer_names]
        layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))
        if 'logits' in layer_names:
            layers.insert(layer_names.index('logits'), self._model.layers[-1])
        assert len(layers) == len(layer_names)
        layer_out_tensors = [layer.output for layer in layers]
        functor = K.function([input_tensor] + [K.learning_phase()], layer_out_tensors)  # evaluate all tensors at once
        layer_outputs = functor([images, 0.])  # 0 to signal testing phase
        return OrderedDict([(layer_name, layer_output) for layer_name, layer_output in zip(layer_names, layer_outputs)])

    def __repr__(self):
        return repr(self._model)

    @property
    def weights(self):
        weights_biases = OrderedDict((layer.name, layer.get_weights()) for layer in self._model.layers)
        weights = []
        for layer, weight_bias in weights_biases.items():
            if not weight_bias:
                continue
            for weight_type, values in zip(["weight", "bias"], weight_bias):
                # reshape into expected (outputs, [inputs, [conv_w, conv_h]])
                if len(values.shape) == 1:  # bias
                    pass
                elif len(values.shape) == 2:  # dense
                    # dense: input, outputs
                    # https://github.com/keras-team/keras/blob/04cbccc8038c105374eef6eb2ce96d6746999860/keras/layers/core.py#L891
                    values = values.transpose(1, 0)
                elif len(values.shape) == 4:  # convolutional
                    # kernel_shape = kernel_size + (input_dim, filters), i.e. 3, 3, 3, 64 = 3x3 convs, RGB, 64 outputs
                    # https://github.com/keras-team/keras/blob/04cbccc8038c105374eef6eb2ce96d6746999860/keras/layers/convolutional.py#L135
                    values = values.transpose(3, 2, 0, 1)
                else:
                    raise ValueError(f"Unknown weight shape {values.shape}")
                values, flatten_coords = collapse_weights(values)
                parameter_weights = DataArray(values, coords={
                    **{'layer': ('weights', [layer] * len(values)),
                       'type': ('weights', [weight_type] * len(values)),
                       },
                    **{coord: ('weights', coord_values) for coord, coord_values in flatten_coords.items()},
                }, dims=['weights'])
                weights.append(parameter_weights)
        weights = merge_weight_assemblies(weights)
        return weights

    @property
    def units(self):
        weight_layers = [layer.name for layer in self._model.layers if layer.get_weights()]
        # run dummy image
        image_filepath = Path(__file__).parent / 'dummy.jpg'
        activations = self.from_paths([image_filepath], layers=weight_layers)
        return DataArray(activations['neuroid'])

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for layer in self._model.layers:
            g.add_node(layer.name, object=layer, type=type(layer))
            for outbound_node in layer._outbound_nodes:
                g.add_edge(layer.name, outbound_node.outbound_layer.name)
        return g


def load_images(image_filepaths, image_size):
    images = [load_image(image_filepath) for image_filepath in image_filepaths]
    images = [scale_image(image, image_size) for image in images]
    return np.array(images)


def load_image(image_filepath):
    from keras.preprocessing import image
    img = image.load_img(image_filepath)
    x = image.img_to_array(img)
    return x


def scale_image(img, image_size):
    from PIL import Image
    from keras.preprocessing import image
    img = Image.fromarray(img.astype(np.uint8))
    img = img.resize((image_size, image_size))
    img = image.img_to_array(img)
    return img


def preprocess(image_filepaths, image_size, *args, **kwargs):
    # only a wrapper to avoid top-level keras imports
    from keras.applications.imagenet_utils import preprocess_input
    images = load_images(image_filepaths, image_size=image_size)
    return preprocess_input(images, *args, **kwargs)
