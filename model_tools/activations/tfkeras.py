from collections import OrderedDict
import numpy as np
from model_tools.activations.core import ActivationsExtractorHelper
import tensorflow as tf


class TFKerasWrapper:
    """
    A wrapper for the Model class created from tensorflow(>=2.0.0).keras
    """
    def __init__(self,
                 model,
                 preprocessing,
                 identifier=None,
                 *args, **kwargs):
        """
        :param model: a tf.keras model with a function `preprocess_input`
            that will later be called on the loaded numpy image
        """
        self._model = model
        identifier = identifier or model.name
        self._extractor = ActivationsExtractorHelper(
            identifier=identifier, get_activations=self.get_activations, preprocessing=preprocessing,
            *args, **kwargs)
        self._extractor.insert_attrs(self)

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def get_activations(self, images, layer_names):

        """
        param images: a list of image paths
        param layer_names: a list of layer names
        """
        from tensorflow.keras import backend as K

        input_tensor = self._model.input
        layers = [layer for layer in self._model.layers if layer.name in layer_names]
        layers = sorted(layers, key=lambda layer: layer_names.index(layer.name))

        if 'logits' in layer_names:
            layers.insert(layer_names.index('logits'), self._model.layers[-1])

        assert len(layers) == len(layer_names)
        layer_out_tensors = [layer.output for layer in layers]
        functor = K.function([input_tensor], layer_out_tensors)  # evaluate all tensors at once
        K.set_learning_phase(0)  # 0 to signal testing phase
        layer_outputs = functor([images])
        return OrderedDict([(layer_name, layer_output) for layer_name, layer_output in zip(layer_names, layer_outputs)])

    def __repr__(self):
        return repr(self._model)

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for layer in self._model.layers:
            g.add_node(layer.name, object=layer, type=type(layer))
            for outbound_node in layer._outbound_nodes:
                g.add_edge(layer.name, outbound_node.outbound_layer.name)
        return g



def tfkeras_load_images(image_paths):
    """
    :param image_paths: list of strings of len B
    return tf.Tensor of [B, H, W, 3] of dtype tf.uint8
    """

    def load_image(path):
        """
        param path: tf.Tensor (1,)
        """
        blob = tf.io.read_file(path[0])
        im = tf.image.decode_png(blob, channels=3)
        return im

    def load_images(paths):
        images = tf.map_fn(load_image, elems=paths, dtype=tf.uint8)
        return images

    images = list(map(lambda s: [s], image_paths))
    images = tf.constant(images)
    images = load_images(images)  # [b, h, w, 3], dtype tf.uint8

    return images


def resnet_preprocessing(image_paths,
                         image_size=224):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

    images = tfkeras_load_images(image_paths)

    # Execute resizing
    images = tf.image.resize(
                            images,
                            (image_size, image_size),
                            method=tf.image.ResizeMethod.BILINEAR,
                            preserve_aspect_ratio=False,
                            antialias=False,
                            )

    images = images - CHANNEL_MEANS

    return images
