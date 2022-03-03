import numpy as np
import onnxruntime
import onnxoptimizer

SUBMODULE_SEPARATOR = '.'

from collections import OrderedDict
from model_tools.activations.core import ActivationsExtractorHelper
from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs
from skl2onnx.helpers.onnx_helper import select_model_inputs_outputs
from skl2onnx.helpers.onnx_helper import save_onnx_model
import onnx
import torch
from torch.autograd import Variable


class OnnxWrapper:
    def __init__(self, model, preprocessing, identifier=None, *args, **kwargs):
        """
        :param model: a keras model with a function `preprocess_input`
            that will later be called on the loaded numpy image
        """
        self._model = model
        identifier = identifier or model.name
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # create directory to store ONNX models
        import os
        if not os.path.exists("ONNX Partial Models"):
            os.makedirs("ONNX Partial Models")

        # check to make sure model is legitimate
        onnx_model = self._model
        model_name = self.identifier
        onnx.checker.check_model(onnx_model)

        # get the layer names and last layer
        output_names = []
        for out in enumerate_model_node_outputs(onnx_model):
            output_names.append(out)
        last_layer = output_names[-1]

        # init activations dict for return:
        new_dict = {}

        # loop through each layer:
        for layer in layer_names:

            # handle logits case - get last layer activations
            if layer_names[0] == 'logits':
                onnx_layer_output = select_model_inputs_outputs(onnx_model, f'{last_layer}')
            else:
                onnx_layer_output = select_model_inputs_outputs(onnx_model, f'{layer}')

            # optimize and save the ONNX model:
            passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
            optimized_model = onnxoptimizer.optimize(onnx_layer_output, passes)

            # some model layer names have / in them, which throw off saving and loading.
            if "/" in layer:
                parsed_layer = layer.replace("/", "-")
            else:
                parsed_layer = layer

            # save the ONNX model layer
            save_onnx_model(optimized_model,
                            f"ONNX Partial Models/{model_name}_layer_{parsed_layer}_output_optimized.onnx")

            # start up ONNX Runtime
            sess = onnxruntime.InferenceSession(f"ONNX Partial Models/{model_name}_layer_{parsed_layer}_output_optimized.onnx")

            # prepare the input
            def to_numpy(tensor):
                return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

            # process images:
            torch_images = [torch.from_numpy(image) for image in images]
            var_images = Variable(torch.stack(torch_images))
            images_device = var_images.to(self._device)

            # compute ONNX Runtime output prediction
            ort_inputs = {sess.get_inputs()[0].name: to_numpy(images_device)}
            ort_outs = sess.run(None, ort_inputs)
            activations = ort_outs[0]

            # add the layer and its activations
            new_dict[layer] = activations

        final_result = OrderedDict(new_dict)
        return final_result

    def __repr__(self):
        return repr(self._model)

    # def graph(self):
    #     import networkx as nx
    #     g = nx.DiGraph()
    #     for layer in self._model.layers:
    #         g.add_node(layer.name, object=layer, type=type(layer))
    #         for outbound_node in layer._outbound_nodes:
    #             g.add_edge(layer.name, outbound_node.outbound_layer.name)
    #     return g


# takes any framework from supported list and converted to ONNX
def to_onnx(batch_size, in_channel, image_size, model, model_name):

    # generate dummy input
    x = torch.randn(batch_size, in_channel, image_size, image_size, requires_grad=True)
    torch_out = model(x)

    # Export the model to onnx
    torch.onnx.export(model,                        # model being run
                      x,                            # model input (or a tuple for multiple inputs)
                      f"{model_name}.onnx",              # where to save the model (can be a file or file-like object)
                      export_params=True,           # store the trained parameter weights inside the model file
                      opset_version=10,             # the ONNX version to export the model to
                      do_constant_folding=True,     # whether to execute constant folding for optimization
                      input_names=['input'],        # the model's input names
                      output_names=['output'],                   # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load( f"{model_name}.onnx")
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession( f"{model_name}.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    return onnx_model


def get_layers(onnx_model):
    layers = []
    for out in enumerate_model_node_outputs(onnx_model):
        layers.append(out)
    return layers


def get_final_model(framework, batch_size, in_channels, image_size, model, model_name):

    # print(batch_size, in_channels, image_size)

    # if model is pytorch, convert to ONNX automatically
    if framework == "pytorch":
        model.eval()
        onnx_model = to_onnx(batch_size, in_channels, image_size, model, model_name)
        layers = get_layers(onnx_model)
        print("Pytorch to ONNX Conversion successful.")
        return onnx_model, layers

    # if model is already onnx, return that.
    elif framework == "onnx":
        onnx_model = model
        layers = get_layers(onnx_model)
        return onnx_model, layers

    # unknown model format. In the future, I hope to add automatic conversion to ONNX for other platforms
    else:
        raise RuntimeError(f"Given framework {framework} not implemented yet. Please convert your "
                           f"{framework} model to ONNX format. You can view how to do this "
                           f"here: https://github.com/onnx/tutorials")
