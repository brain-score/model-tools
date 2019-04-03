import pytest
import functools
import numpy as np
from os import path
import pickle as pkl
from pytest import approx

from model_tools.activations import PytorchWrapper
from model_tools.brain_transformation.temporal_map import TemporalModelCommitment

# from brainscore.benchmarks.loaders import DicarloMajaj2015TemporalITLowvarLoader, DicarloMajaj2015ITHighvarLoader, \
# 	DicarloMajaj2015TemporalITHighvarLoader

def load_test_assemblies(variation, region):
	if type(variation) is not list:
		variation = [variation]
	assembly_dir = path.join(path.dirname(__file__), 'test_temporal_assemblies')
	test_assembly_filename = 'temporal_test_{}_var{}.pkl'.format(region, ''.join(str(v) for v in variation))
	load_file = path.join(assembly_dir, test_assembly_filename)
	with open(load_file, "rb") as fh:
		test_assembly = pkl.load(fh)
	return test_assembly

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

class TestTemporalModelCommitment:
	test_data = [(pytorch_custom, ['linear', 'relu2'], 'test_temporal', 'IT')]
	@pytest.mark.parametrize("model_ctr, layers, identifier, region", test_data)
	def test(self, model_ctr, layers, identifier, region):
		# train_assembly_loader = DicarloMajaj2015TemporalITLowvarLoader()
		# test_assembly_loader = DicarloMajaj2015TemporalITHighvarLoader()
		# commit_loader = DicarloMajaj2015ITHighvarLoader()

		training_assembly = load_test_assemblies([0,3], region)
		validation_assembly = load_test_assemblies(6, region)
		# commit_assembly = commit_loader()

		expected_region = region if type(region)==list else [region]
		expected_region_count = len(expected_region)
		expected_time_bin_count = len(training_assembly.time_bin.values)

		extractor = pytorch_custom()

		t_bins = [t for t in training_assembly.time_bin.values if t[0] >= 0]
		expected_recorded_time_count = len(t_bins)

		temporal_model = TemporalModelCommitment(identifier, extractor, layers)
		# commit region:
		temporal_model.commit_region(region, validation_assembly)
		temporal_model.do_commit_region(region)
		# make temporal:
		temporal_model.make_temporal(training_assembly)
		assert len(temporal_model._temporal_maps.keys()) == expected_region_count
		assert len(temporal_model._temporal_maps[region].keys()) == expected_time_bin_count
		# start recording:
		temporal_model.start_recording(region, t_bins)
		assert temporal_model.recorded_regions == expected_region
		# look at:
		stim = validation_assembly.stimulus_set
		temporal_activations = temporal_model.look_at(stim)
		assert set(temporal_activations.region.values) == set(expected_region)
		assert len(set(temporal_activations.time_bin.values)) == expected_recorded_time_count
