import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image
from xarray import DataArray

from model_tools.activations.core import ActivationsExtractorHelper, collapse_weights, merge_weight_assemblies
from model_tools.utils import fullname

SUBMODULE_SEPARATOR = '.'


class PytorchWrapper:
    def __init__(self, model, preprocessing, identifier=None, *args, **kwargs):
        import torch
        logger = logging.getLogger(fullname(self))
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(f"Using device {self._device}")
        self._model = model
        self._model = self._model.to(self._device)
        identifier = identifier or model.__class__.__name__
        self._extractor = self._build_extractor(
            identifier=identifier, preprocessing=preprocessing, get_activations=self.get_activations, *args, **kwargs)
        self._extractor.insert_attrs(self)

    def _build_extractor(self, identifier, preprocessing, get_activations, *args, **kwargs):
        return ActivationsExtractorHelper(
            identifier=identifier, get_activations=get_activations, preprocessing=preprocessing,
            *args, **kwargs)

    @property
    def identifier(self):
        return self._extractor.identifier

    @identifier.setter
    def identifier(self, value):
        self._extractor.identifier = value

    def __call__(self, *args, **kwargs):  # cannot assign __call__ as attribute due to Python convention
        return self._extractor(*args, **kwargs)

    def get_activations(self, images, layer_names):
        import torch
        from torch.autograd import Variable
        images = [torch.from_numpy(image) for image in images]
        images = Variable(torch.stack(images))
        images = images.to(self._device)
        self._model.eval()

        layer_results = OrderedDict()
        hooks = []

        for layer_name in layer_names:
            layer = self.get_layer(layer_name)
            hook = self.register_hook(layer, layer_name, target_dict=layer_results)
            hooks.append(hook)

        self._model(images)
        for hook in hooks:
            hook.remove()
        return layer_results

    def get_layer(self, layer_name):
        if layer_name == 'logits':
            return self._output_layer()
        module = self._model
        for part in layer_name.split(SUBMODULE_SEPARATOR):
            module = module._modules.get(part)
            assert module is not None, f"No submodule found for layer {layer_name}, at part {part}"
        return module

    def _output_layer(self):
        module = self._model
        while module._modules:
            module = module._modules[next(reversed(module._modules))]
        return module

    @classmethod
    def _tensor_to_numpy(cls, output):
        return output.cpu().data.numpy()

    def register_hook(self, layer, layer_name, target_dict):
        def hook_function(_layer, _input, output, name=layer_name):
            target_dict[name] = PytorchWrapper._tensor_to_numpy(output)

        hook = layer.register_forward_hook(hook_function)
        return hook

    def __repr__(self):
        return repr(self._model)

    def layers(self):
        for name, module in self._model.named_modules():
            if len(list(module.children())) > 0:  # this module only holds other modules
                continue
            yield name, module

    _weight_types = ['weight', 'bias']

    @property
    def weights(self):
        state_dict = self._model.state_dict()
        # conv: out_channels, in_channels, *kernel_size
        # (e.g. for 2, 3, 3, 3: 2 outputs, 3 RGB, 3x3 convolutions)
        # (e.g. for 2, 3, 5, 5: 2 outputs, 3 RGB, 5x5 convolutions)
        # bias: out_channels (2 for conv1)
        weights = []
        for name, tensor in state_dict.items():
            values, flatten_coords = collapse_weights(self._tensor_to_numpy(tensor))
            weight_type = [t for t in self._weight_types if name.endswith(f".{t}")][0]
            layer = name.rstrip(f".{weight_type}")
            parameter_weights = DataArray(values, coords={
                **{'parameter': ('weights', [name] * len(values)),
                   'type': ('weights', [weight_type] * len(values)),
                   'layer': ('weights', [layer] * len(values)),
                   },
                **{coord: ('weights', coord_values) for coord, coord_values in flatten_coords.items()},
            }, dims=['weights'])
            weights.append(parameter_weights)
        weights_assembly = merge_weight_assemblies(weights)
        return weights_assembly

    @property
    def units(self):
        # count only units in layers with weights
        parameters = list(self._model.state_dict().keys())
        weight_layers = []
        for param in parameters:
            for _rstrip in self._weight_types:
                param = param.rstrip(f".{_rstrip}")
            weight_layers.append(param)
        weight_layers = set(weight_layers)
        # run dummy image
        image_filepath = Path(__file__).parent / 'dummy.jpg'
        activations = self.from_paths([image_filepath], layers=weight_layers)
        return DataArray(activations['neuroid'])

    def graph(self):
        import networkx as nx
        g = nx.DiGraph()
        for layer_name, layer in self.layers():
            g.add_node(layer_name, object=layer, type=type(layer))
        return g


def load_preprocess_images(image_filepaths, image_size):
    images = load_images(image_filepaths)
    images = preprocess_images(images, image_size=image_size)
    return images


def load_images(image_filepaths):
    return [load_image(image_filepath) for image_filepath in image_filepaths]


def load_image(image_filepath):
    with Image.open(image_filepath) as pil_image:
        if 'L' not in pil_image.mode.upper() and 'A' not in pil_image.mode.upper():  # not binary and not alpha
            # work around to https://github.com/python-pillow/Pillow/issues/1144,
            # see https://stackoverflow.com/a/30376272/2225200
            return pil_image.copy()
        else:  # make sure potential binary images are in RGB
            rgb_image = Image.new("RGB", pil_image.size)
            rgb_image.paste(pil_image)
            return rgb_image


def preprocess_images(images, image_size):
    preprocess = torchvision_preprocess_input(image_size)
    images = [preprocess(image) for image in images]
    images = np.concatenate(images)
    return images


def torchvision_preprocess_input(image_size):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(image_size),
        torchvision_preprocess(),
    ])


def torchvision_preprocess(normalize_mean=(0.485, 0.456, 0.406), normalize_std=(0.229, 0.224, 0.225)):
    from torchvision import transforms
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
        lambda img: img.unsqueeze(0)
    ])
