import functools
import os

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from model_tools.brain_transformation import ModelCommitment
from model_tools.activations import PytorchWrapper
import brainscore
import brainio_collection
from brainscore.model_interface import BrainModel

def pytorch_custom(image_size):
    import torch
    from torch import nn
    from model_tools.activations.pytorch import load_preprocess_images

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, bias=False)
            self.relu1 = torch.nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu1(x)
            return x

    preprocessing = functools.partial(load_preprocess_images, image_size=image_size)
    return PytorchWrapper(model=MyModel(), preprocessing=preprocessing)

class TestObjectSearch:
    def test_model(self):
        target_model_pool = pytorch_custom(28)
        stimuli_model_pool = pytorch_custom(224)
        search_target_model_param = {}
        search_stimuli_model_param = {}
        search_target_model_param['target_model'] = target_model_pool
        search_stimuli_model_param['stimuli_model'] = stimuli_model_pool
        search_target_model_param['target_layer'] = 'relu1'
        search_stimuli_model_param['stimuli_layer'] = 'relu1'
        search_target_model_param['target_img_size'] = 28
        search_stimuli_model_param['search_image_size'] = 224

        model = ModelCommitment(identifier=stimuli_model_pool.identifier, activations_model=None, layers=['relu1'], search_target_model_param=search_target_model_param, search_stimuli_model_param=search_stimuli_model_param)
        assemblies = brainscore.get_assembly('klab.Zhang2018search_obj_array')
        stimuli = assemblies.stimulus_set
        fix = [[640, 512],
               [365, 988],
               [90, 512],
               [365, 36],
               [915, 36],
               [1190, 512],
               [915, 988]]
        max_fix = 6
        data_len = 300
        model.start_task(BrainModel.Task.visual_search_obj_arr, fix=fix, max_fix=max_fix, data_len=data_len)
        cumm_perf, saccades = model.look_at(stimuli)

        assert saccades.shape == (300, 8, 2)
        assert cumm_perf.shape == (7, 2)
