import functools

import numpy as np

from model_tools.activations import PytorchWrapper, KerasWrapper, TensorflowSlimWrapper


def pytorch_custom():
    import torch
    from torch import nn
    from model_tools.activations.pytorch import load_preprocess_images

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3)
            self.relu1 = torch.nn.ReLU()
            linear_input_size = np.power((224 - 3 + 2 * 0) / 1 + 1, 2) * 2
            self.linear = torch.nn.Linear(int(linear_input_size), 1000)
            self.relu2 = torch.nn.ReLU()  # can't get named ReLU output otherwise

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = self.relu2(x)
            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return PytorchWrapper(model=MyModel(), preprocessing=preprocessing)


def pytorch_alexnet():
    from torchvision.models.alexnet import alexnet
    from model_tools.activations.pytorch import load_preprocess_images

    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    return PytorchWrapper(model=alexnet(pretrained=True), preprocessing=preprocessing)


def pytorch_alexnet_resize():
    from torchvision.models.alexnet import alexnet
    from model_tools.activations.pytorch import load_images, torchvision_preprocess
    from torchvision import transforms
    torchvision_preprocess_input = transforms.Compose([transforms.Resize(224), torchvision_preprocess()])

    def preprocessing(paths):
        images = load_images(paths)
        images = [torchvision_preprocess_input(image) for image in images]
        images = np.concatenate(images)
        return images

    return PytorchWrapper(alexnet(pretrained=True), preprocessing, identifier='alexnet-resize')


def keras_vgg19():
    import keras
    from keras.applications.vgg19 import VGG19, preprocess_input
    from model_tools.activations.keras import load_images
    keras.backend.clear_session()
    preprocessing = lambda image_filepaths: preprocess_input(load_images(image_filepaths, image_size=224))
    return KerasWrapper(model=VGG19(), preprocessing=preprocessing)


def tfslim_custom():
    from model_tools.activations.tensorflow import load_resize_image
    import tensorflow as tf
    slim = tf.contrib.slim
    tf.reset_default_graph()

    image_size = 224
    placeholder = tf.placeholder(dtype=tf.string, shape=[64])
    preprocess = lambda image_path: load_resize_image(image_path, image_size)
    preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)

    with tf.variable_scope('my_model', values=[preprocess]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # Collect outputs for conv2d, fully_connected and max_pool2d.
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                            outputs_collections=[end_points_collection]):
            net = slim.conv2d(preprocess, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [5, 5], 5, scope='pool1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.flatten(net, scope='flatten')
            net = slim.fully_connected(net, 1000, scope='logits')
            endpoints = slim.utils.convert_collection_to_dict(end_points_collection)

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    return TensorflowSlimWrapper(identifier='tf-custom', labels_offset=0,
                                 endpoints=endpoints, inputs=placeholder, session=session)


def tfslim_vgg16():
    import tensorflow as tf
    from nets import nets_factory
    from preprocessing import vgg_preprocessing
    from model_tools.activations.tensorflow import load_resize_image
    tf.reset_default_graph()

    image_size = 224
    placeholder = tf.placeholder(dtype=tf.string, shape=[64])
    preprocess_image = lambda image: vgg_preprocessing.preprocess_image(
        image, image_size, image_size, resize_side_min=image_size)
    preprocess = lambda image_path: preprocess_image(load_resize_image(image_path, image_size))
    preprocess = tf.map_fn(preprocess, placeholder, dtype=tf.float32)

    model_ctr = nets_factory.get_network_fn('vgg_16', num_classes=1001, is_training=False)
    logits, endpoints = model_ctr(preprocess)

    session = tf.Session()
    session.run(tf.initialize_all_variables())
    return TensorflowSlimWrapper(identifier='tf-vgg16', labels_offset=1,
                                 endpoints=endpoints, inputs=placeholder, session=session)