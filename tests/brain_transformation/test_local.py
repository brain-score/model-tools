import functools
import numpy as np
import os
import pytest
import xarray as xr
from pathlib import Path
from pytest import approx

from brainio.assemblies import BehavioralAssembly
from brainio.stimuli import StimulusSet
from brainscore.benchmarks.rajalingham2018 import _DicarloRajalingham2018
from brainscore.benchmarks.screen import place_on_screen
from brainscore.metrics.image_level_behavior import I2n

import sys
file_path = "/Users/linussommer/Documents/GitHub/brain-score/brainscore"
sys.path.append(file_path)
import brainscore
from model_interface import BrainModel


import sys
file_path = "/Users/linussommer/Documents/GitHub/model-tools"
sys.path.append(file_path)
from model_tools.activations import PytorchWrapper
from model_tools.brain_transformation import ModelCommitment, ProbabilitiesMapping


def pytorch_custom():
    import torch
    from torch import nn
    from model_tools.activations.pytorch import load_preprocess_images

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            np.random.seed(0)
            torch.random.manual_seed(0)
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

class TestOddOneOut:
    def test_odd_one_out(self):
        activations_model = pytorch_custom()
        brain_model = ModelCommitment(identifier=activations_model.identifier, activations_model=activations_model,
                                      layers=[None], behavioral_readout_layer='relu2')
                
        assy = brainscore.get_assembly(f'Hebart2023')

        idx1 = assy.coords["image_1"].values
        idx2 = assy.coords["image_2"].values
        idx3 = assy.coords["image_3"].values
        triplets = np.array([idx1, idx2, idx3]).T

        fitting_stimuli = assy.stimulus_set
        fitting_stimuli = place_on_screen(fitting_stimuli, target_visual_degrees=brain_model.visual_degrees(),
                                            source_visual_degrees=8) 
        brain_model.start_task(BrainModel.Task.odd_one_out)

        data = [fitting_stimuli, triplets]
        choices = brain_model.look_at(data)

        n = 0
        for i in range(len(choices)):
            if choices[i] == idx3[i]:
                n += 1

        print(n, n/len(choices))

test = TestOddOneOut()
test.test_odd_one_out() 