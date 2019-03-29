import pytest
import functools
import numpy as np
from pytest import approx

from model_tools.activations import PytorchWrapper
from model_tools.brain_transformation.temporal_map import TemporalModelCommitment

from brainscore import get_assembly
from brainscore.benchmarks.loaders import AssemblyLoader, DicarloMajaj2015Loader, DicarloMajaj2015ITLoader

# create a test assembly:
class DicarloMajaj2015TemporalLoader(AssemblyLoader):                          # needed to add variation argument
	def __init__(self, name='dicarlo.Majaj2015.temporal'):
		super(DicarloMajaj2015TemporalLoader, self).__init__(name=name)
		self._helper = DicarloMajaj2015Loader()

	def __call__(self, average_repetition=True, variation=6):
		assembly = get_assembly(name='dicarlo.Majaj2015.temporal')
		assembly = self._helper._filter_erroneous_neuroids(assembly)
		assembly = assembly.sel(variation=variation)
		assembly = assembly.transpose('presentation', 'neuroid', 'time_bin')
		if average_repetition:
			assembly = self._helper.average_repetition(assembly)
		return assembly

def get_stim(assembly):
	return assembly.stimulus_set[assembly.stimulus_set['image_id'].isin(assembly['image_id'].values)]

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
	@pytest.mark.parametrize(['model_ctr', 'layers', 'expected_best_layer', 'region'
							  , 'expected_region_count', 'expected_recorded_regions'
							  , 'expected_time_bin_count', 'expected_recorded_time_bin_cnt'
                              , 'expected_time_bins', 'expected_recorded_time_bins'],
							 [pytorch_custom, ['linear', 'relu2'], 'relu2', 'IT'
								    , 1, ['IT'], 39, 29
                                    ,[(-100, -80),(-90, -70),(-80, -60),(-70, -50),(-60, -40)
                                        ,(-50, -30),(-40, -20),(-30, -10),(-20, 0),(-10, 10),(0, 20),(10, 30),(20, 40)
                                        ,(30, 50),(40, 60),(50, 70),(60, 80),(70, 90),(80, 100),(90, 110),(100, 120)
                                        ,(110, 130),(120, 140),(130, 150),(140, 160),(150, 170),(160, 180),(170, 190)
                                        ,(180, 200),(190, 210),(200, 220),(210, 230),(220, 240),(230, 250),(240, 260)
                                        ,(250, 270),(260, 280),(270, 290),(280, 300)]
                                    , [(0, 20),(10, 30),(20, 40)
                                        ,(30, 50),(40, 60),(50, 70),(60, 80),(70, 90),(80, 100),(90, 110),(100, 120)
                                        ,(110, 130),(120, 140),(130, 150),(140, 160),(150, 170),(160, 180),(170, 190)
                                        ,(180, 200),(190, 210),(200, 220),(210, 230),(220, 240),(230, 250),(240, 260)
                                        ,(250, 270),(260, 280),(270, 290),(280, 300)]])
	def test(self, model_ctr, layers, expected_best_layer, region
			, expected_region_count, expected_recorded_regions, expected_time_bin_count, expected_recorded_time_bin_cnt
			 , expected_time_bins, expected_recorded_time_bins):
		train_test_assembly_loader = DicarloMajaj2015TemporalLoader()
		commit_loader = DicarloMajaj2015ITLoader()

		training_assembly = train_test_assembly_loader(variation=3)
		commit_assembly = commit_loader(average_repetition=False)
		validation_assembly = train_test_assembly_loader(variation=6)

		extractor = pytorch_custom()

		t_bins = [t for t in training_assembly.time_bin.values if t[0] >= 0]

		temporal_model = TemporalModelCommitment(extractor, layers)
		# commit region:
		temporal_model.commit_region(region, commit_assembly)
		assert temporal_model.region_layer_map[region] == expected_best_layer
		# make temporal:
		temporal_model.make_temporal(training_assembly)
		assert len(temporal_model._temporal_maps.keys()) == expected_region_count
		assert len(temporal_model._temporal_maps[region].keys()) == expected_time_bin_count
		assert set(temporal_model._temporal_maps[region].keys()) == expected_time_bins
		# start recording:
		temporal_model.start_temporal_recording(region, t_bins)
		assert temporal_model.recorded_regions == expected_recorded_regions
		# look at:
		stim = get_stim(commit_assembly)
		temporal_activations = temporal_model.look_at(stim)
		assert set(temporal_activations.region.values) == set(expected_recorded_regions)
		assert len(set(temporal_activations.time_bin.values)) == expected_recorded_time_bin_cnt
		assert set(temporal_activations.time_bin.values) == expected_recorded_time_bins
