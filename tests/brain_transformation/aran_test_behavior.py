import functools
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytest import approx

from brainio_base.stimuli import StimulusSet
from brainscore.metrics.behavior import I2n
from brainscore.metrics.transformations import subset
from brainscore.model_interface import BrainModel
from model_tools.brain_transformation import ProbabilitiesMapping
from candidate_models.base_models import base_model_pool

def get_objectome(subtype):
    basepath = '/home/anayebi/Rajalingham2018/'
    with open(f'{basepath}/dicarlo.Rajalingham2018.{subtype}.pkl',
              'rb') as f:
        objectome = pickle.load(f)
    with open(f'{basepath}/dicarlo.Rajalingham2018.{subtype}-stim.pkl',
              'rb') as f:
        stimulus_set = pickle.load(f)
    objectome.attrs['stimulus_set'] = stimulus_set
    return objectome

fitting_objectome, testing_objectome = get_objectome('partial_trials'), get_objectome('full_trials')

activations_model = base_model_pool['resnet-101_v2']
# transform
transformation = ProbabilitiesMapping(identifier='TestI2Nresnet101_v2',
                                      activations_model=activations_model,
                                      layer='global_pool')
transformation.start_task(BrainModel.Task.probabilities, fitting_objectome.stimulus_set)
testing_features = transformation.look_at(testing_objectome.stimulus_set)
# metric
i2n = I2n()
score = i2n(testing_features, testing_objectome)
score = score.sel(aggregation='center')
print(score)
