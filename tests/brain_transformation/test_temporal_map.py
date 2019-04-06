import pytest
import functools
import numpy as np
from os import path

from model_tools.activations import PytorchWrapper
from model_tools.brain_transformation.temporal_map import TemporalModelCommitment

from brainscore.metrics.regression import pls_regression
from brainscore.assemblies.public import load_assembly

from xarray import DataArray
from pandas import DataFrame

from brainio_base.stimuli import StimulusSet
from brainio_base.assemblies import NeuronRecordingAssembly

def load_test_assemblies(variation, region):
	image_dir = path.join(path.dirname(path.abspath(__file__)), 'test_temporal_stimulus')
	if type(variation) is not list:
		variation = [variation]

	num_stim = 5
	neuroid_cnt = 168
	time_bin_cnt = 5
	resp = np.random.rand(num_stim, neuroid_cnt, time_bin_cnt)

	dims = ['presentation', 'neuroid', 'time_bin']
	coords = {
					'image_id': ('presentation', range(num_stim)),
					'y': ('presentation', range(num_stim)),
					'neuroid_id': ('neuroid', [f'{i}' for i in range(neuroid_cnt)]),
					'region': ('neuroid', ['IT'] * neuroid_cnt),
					'x': ('neuroid', range(neuroid_cnt)),
					'time_bin_start': ('time_bin', range(-10, 40, 10)),
					'time_bin_end': ('time_bin', range(0, 50, 10))
				}

	assembly = DataArray(data=resp, dims=dims, coords=coords)
	assembly = assembly.set_index(presentation=['image_id', 'y'],
	                              neuroid=['neuroid_id','region', 'x'],
	                              time_bin=['time_bin_start', 'time_bin_end'],
	                              append=True)

	stim_meta = [{'id': k} for k in range(num_stim)]
	image_paths = {}
	for i in range(num_stim):
		f_name = f"im_{i:05}.jpg"
		im_path = path.join(image_dir, f_name)

		meta = stim_meta[i]
		meta['image_id'] = f'{i}'
		meta['image_file_name'] = f_name
		image_paths[f'{i}'] = im_path

	stim_set = DataFrame(stim_meta)

	stim_set = StimulusSet(stim_set)
	stim_set.image_paths = image_paths
	stim_set.name = f'testing_temporal_stims_{region}_var{"".join(str(v) for v in variation)}'

	assembly = NeuronRecordingAssembly(assembly)

	assembly.attrs['stimulus_set'] = stim_set
	assembly.attrs['stimulus_set_name'] = stim_set.name
	return assembly

def pytorch_custom():
	import torch
	from torch import nn
	from model_tools.activations.pytorch import load_preprocess_images

	class MyModel_Temporal(nn.Module):
		def __init__(self):
			super(MyModel_Temporal, self).__init__()
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
	return PytorchWrapper(model=MyModel_Temporal(), preprocessing=preprocessing)

class TestTemporalModelCommitment:
	test_data = [(pytorch_custom, ['linear', 'relu2'], 'IT')]
	@pytest.mark.parametrize("model_ctr, layers, region", test_data)
	def test(self, model_ctr, layers, region):
		commit_assembly = load_assembly(name='dicarlo.Majaj2015.lowvar.IT',
		                                **{'average_repetition': False})

		training_assembly = load_test_assemblies([0,3], region)
		validation_assembly = load_test_assemblies(6, region)

		expected_region = region if type(region)==list else [region]
		expected_region_count = len(expected_region)
		expected_time_bin_count = len(training_assembly.time_bin.values)

		extractor = model_ctr()

		t_bins = [t for t in training_assembly.time_bin.values if 0 <= t[0] < 30]
		expected_recorded_time_count = len(t_bins)

		temporal_model = TemporalModelCommitment('', extractor, layers)
		# commit region:
		temporal_model.commit_region(region, commit_assembly)
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
		#
		test_layer = temporal_model.region_layer_map[region]
		train_stim_set = training_assembly.stimulus_set
		for time_test in t_bins:
			target_assembly = training_assembly.sel(time_bin=time_test, region=region)
			region_activations = extractor(train_stim_set, [test_layer])
			regressor = pls_regression(neuroid_coord=('neuroid_id', 'layer', 'region'))
			regressor.fit(region_activations, target_assembly)
			#
			test_activations = extractor(stim, [test_layer])
			test_predictions = regressor.predict(test_activations).values
			#
			temporal_model_prediction = temporal_activations.sel(region=region, time_bin=time_test).values
			assert temporal_model_prediction == pytest.approx(test_predictions, rel=1e-3, abs=1e-6)

