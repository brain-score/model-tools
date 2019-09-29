import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
from xarray import DataArray

from model_tools.activations.core import ActivationsExtractorHelper, collapse_weights, merge_weight_assemblies

_logger = logging.getLogger(__name__)


class TensorflowWrapper:
    def __init__(self, identifier, inputs, endpoints: dict, session, *args, **kwargs):
        import tensorflow as tf
        self._inputs = inputs
        self._endpoints = endpoints
        self._session = session or tf.Session()
        self._extractor = ActivationsExtractorHelper(identifier=identifier, get_activations=self.get_activations,
                                                     preprocessing=None, *args, **kwargs)
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
        layer_tensors = OrderedDict((layer, self._endpoints[
            layer if (layer != 'logits' or layer in self._endpoints) else next(reversed(self._endpoints))])
                                    for layer in layer_names)
        layer_outputs = self._session.run(layer_tensors, feed_dict={self._inputs: images})
        return layer_outputs

    @property
    def weights(self):
        import tensorflow as tf
        weights = []
        for variable in tf.trainable_variables():
            values = self._session.run(variable)  # shape: kernel_x x kernel_y x input x output
            # transform into outputs, inputs, kernel_x, kernel_y
            if len(values.shape) == 4:  # convolutional
                values = np.transpose(values, [3, 2, 0, 1])
            elif len(values.shape) == 2:  # dense
                values = np.transpose(values, [1, 0])
            elif len(values.shape) == 1:  # bias
                pass
            else:
                raise ValueError(f"Unknown variable shape length: {values.shape}")
            values, flatten_coords = collapse_weights(values)
            parameter_weights = DataArray(values, coords={
                **{'name': ('weights', [variable.name] * len(values)),
                   },
                **{coord: ('weights', coord_values) for coord, coord_values in flatten_coords.items()},
            }, dims=['weights'])
            weights.append(parameter_weights)
        weights = merge_weight_assemblies(weights)
        return weights

    def list_ops(self):
        import tensorflow as tf
        include_prefix = "/"
        exclude_prefix = include_prefix + "_"
        ops = tf.get_default_graph().get_operations()
        # ops = [op for op in ops if op.name.startswith(include_prefix)]
        ops = [op for op in ops]
        ops = [op for op in ops if not op.name.startswith(exclude_prefix)]
        return ops

    def _list_layers(self):
        """Returns a list of (layer_name, output_expr, trainable_vars) tuples corresponding to
        individual layers of the network. Mainly intended to be used for reporting."""
        import tensorflow as tf
        layers = []

        def recurse(scope, parent_ops, parent_vars, level):
            # Ignore specific patterns.
            if any(p in scope for p in ["/Shape", "/strided_slice", "/Cast", "/concat", "/Assign"]):
                return

            # Filter ops and vars by scope.
            global_prefix = (scope + "/") if scope else ''
            cur_ops = [op for op in parent_ops if op.name.startswith(global_prefix) or op.name == global_prefix[:-1]]
            cur_vars = [(name, var) for name, var in parent_vars
                        if name.startswith(global_prefix) or name == global_prefix[:-1]]
            if not cur_ops and not cur_vars:
                return

            # Filter out all ops related to variables.
            for var in [op for op in cur_ops if op.type.startswith("Variable")]:
                var_prefix = var.name + "/"
                cur_ops = [op for op in cur_ops if not op.name.startswith(var_prefix)]

            # Filter out init
            cur_ops = [op for op in cur_ops if op.type != 'NoOp']

            # Scope does not contain ops as immediate children => recurse deeper.
            contains_direct_ops = any(
                "/" not in op.name[len(global_prefix):] and op.type != "Identity" for op in cur_ops)
            if (level == 0 or not contains_direct_ops) and (len(cur_ops) + len(cur_vars)) > 1:
                visited = set()
                for rel_name in [op.name[len(global_prefix):] for op in cur_ops] + \
                                [name[len(global_prefix):] for name, _var in cur_vars]:
                    token = rel_name.split("/")[0]
                    if token not in visited:
                        recurse(global_prefix + token, cur_ops, cur_vars, level + 1)
                        visited.add(token)
                return

            # Report layer.
            layer_name = scope
            layer_output = cur_ops[-1].outputs[0] if cur_ops else cur_vars[-1][1]
            layer_trainables = [var for _name, var in cur_vars if var.trainable]
            layers.append((layer_name, layer_output, layer_trainables))

        vars = OrderedDict((var.name, var) for var in tf.global_variables())
        recurse('', self.list_ops(), list(vars.items()), 0)
        return layers

    @property
    def units(self):
        weight_tensors = {layer_name: layer_output.name
                          for layer_name, layer_output, layer_trainables in self._list_layers()
                          if sum(np.prod(var.shape) for var in layer_trainables) > 0}
        tensor_layers = {tensor.name: layer for layer, tensor in self._endpoints.items()}
        weight_layers = []
        for tensor_name in weight_tensors.values():
            if tensor_name not in tensor_layers:
                _logger.warning(f"{tensor_name} not found in endpoints")
                continue
            layer = tensor_layers[tensor_name]
            weight_layers.append(layer)
        # run dummy image
        image_filepath = Path(__file__).parent / 'dummy.jpg'
        activations = self.from_paths([str(image_filepath)], layers=weight_layers)
        return DataArray(activations['neuroid'])

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for name, layer in self._endpoints.items():
            g.add_node(name, object=layer, type=type(layer))
        g.add_node("logits", object=self.logits, type=type(self.logits))
        return g


class TensorflowSlimWrapper(TensorflowWrapper):
    def __init__(self, *args, labels_offset=1, **kwargs):
        super(TensorflowSlimWrapper, self).__init__(*args, **kwargs)
        self._labels_offset = labels_offset

    def get_activations(self, images, layer_names):
        layer_outputs = super(TensorflowSlimWrapper, self).get_activations(images, layer_names)
        if 'logits' in layer_outputs:
            layer_outputs['logits'] = layer_outputs['logits'][:, self._labels_offset:]
        return layer_outputs


def load_image(image_filepath):
    import tensorflow as tf
    image = tf.read_file(image_filepath)
    image = tf.image.decode_png(image, channels=3)
    return image


def resize_image(image, image_size):
    import tensorflow as tf
    image = tf.image.resize_images(image, (image_size, image_size))
    return image


def load_resize_image(image_path, image_size):
    image = load_image(image_path)
    image = resize_image(image, image_size)
    return image
