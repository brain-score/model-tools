import logging
import os
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from brainio_base.stimuli import StimulusSet
from brainio_base.assemblies import merge_data_arrays
from brainscore.benchmarks import BenchmarkBase, ceil_score
from brainscore.benchmarks.loaders import average_repetition
from brainscore.metrics import Score
from brainscore.metrics.ceiling import InternalConsistency
from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression
from brainscore.model_interface import BrainModel
from brainscore.utils import fullname
from result_caching import store_xarray, store


class ModelCommitment(BrainModel):
	def __init__(self, identifier, base_model, layers):
		self.layers = layers

		self.layer_model = LayerModel(identifier=identifier, base_model=base_model)
		# forward brain-interface methods
		self.look_at = self.layer_model.look_at
		self.start_recording = self.layer_model.start_recording
		self.start_task = self.layer_model.start_task

	def commit_region(self, region, assembly):
		layer_selection = LayerSelection(model_identifier=self.layer_model.identifier,
										 activations_model=self.layer_model.base_model, layers=self.layers)
		best_layer = layer_selection(assembly)
		self.layer_model.commit(region, best_layer)


class PixelsToDegrees:
	CENTRAL_VISION_DEGREES = 10

	def __init__(self, target_pixels, target_degrees=CENTRAL_VISION_DEGREES):
		self.target_pixels = target_pixels
		self.target_degrees = target_degrees
		framework_home = os.path.expanduser(os.getenv('MT_HOME', '~/.model-tools'))
		self._directory = os.path.join(framework_home, "stimuli-degrees")

	def __call__(self, stimuli):
		target_dir = os.path.join(self._directory, stimuli.name,
								  f"target_{self.target_degrees}deg_{self.target_pixels}pix")
		os.makedirs(target_dir, exist_ok=True)
		image_paths = {image_id: self.convert_image(stimuli.get_image(image_id),
													image_degrees=degrees, target_dir=target_dir)
					   for image_id, degrees in zip(stimuli['image_id'], stimuli['degrees'])}
		converted_stimuli = StimulusSet(stimuli)  # .copy() for some reason keeps the link to the old metadata
		converted_stimuli.name = f"{stimuli.name}-{self.target_degrees}degrees_{self.target_pixels}"
		converted_stimuli['degrees'] = self.target_degrees
		converted_stimuli.image_paths = image_paths
		converted_stimuli.original_paths = {converted_stimuli.image_paths[image_id]: stimuli.image_paths[image_id]
											for image_id in stimuli['image_id']}
		return converted_stimuli

	@classmethod
	def hook(cls, activations_extractor, target_pixels, target_degrees=CENTRAL_VISION_DEGREES):
		hook = PixelsToDegrees(target_pixels=target_pixels, target_degrees=target_degrees)
		handle = activations_extractor.register_stimulus_set_hook(hook)
		return handle

	def convert_image(self, image_path, image_degrees, target_dir):
		target_path = os.path.join(target_dir, os.path.basename(image_path))
		if not os.path.isfile(target_path):
			image = self._load_image(image_path)
			pixels_per_degree = self.target_pixels / self.target_degrees
			stimulus_pixels = self._round(image_degrees * pixels_per_degree)
			image = self._resize_image(image, image_size=stimulus_pixels)
			image = self._center_on_background(image, background_size=self.target_pixels)
			self._write(image, target_path=target_path)
			image.close()
		return target_path

	def _round(self, number):
		return np.array(number).round().astype(int)

	def _load_image(self, image_path):
		return Image.open(image_path)

	def _resize_image(self, image, image_size):
		return image.resize((image_size, image_size), Image.ANTIALIAS)

	def _center_on_background(self, center_image, background_size, background_color='gray'):
		image = Image.new('RGB', (background_size, background_size), background_color)
		center_topleft = self._round(np.subtract(background_size, center_image.size) / 2)
		image.paste(center_image, tuple(center_topleft))
		return image

	def _write(self, image, target_path):
		image.save(target_path)


class LayerModel(BrainModel):
	def __init__(self, identifier, base_model, region_layer_map: Optional[dict] = None):
		self.identifier = identifier
		self.base_model = base_model
		self.region_layer_map = region_layer_map or {}
		self.recorded_regions = []

	def look_at(self, stimuli):
		layer_regions = {self.region_layer_map[region]: region for region in self.recorded_regions}
		assert len(layer_regions) == len(self.recorded_regions), f"duplicate layers for {self.recorded_regions}"
		activations = self.base_model(stimuli, layers=list(layer_regions.keys()))
		activations['region'] = 'neuroid', [layer_regions[layer] for layer in activations['layer'].values]
		return activations

	def start_task(self, task):
		if task != BrainModel.Task.passive:
			raise NotImplementedError()

	def start_recording(self, recording_target):
		if str(recording_target) not in self.region_layer_map:
			raise NotImplementedError(f"Region {recording_target} is not committed")
		self.recorded_regions = [recording_target]

	def commit(self, region, layer):
		self.region_layer_map[region] = layer


class LayerSelection:
	def __init__(self, model_identifier, activations_model, layers):
		"""
		:param model_identifier: this is separate from the container name because it only refers to
			the combination of (model, preprocessing), i.e. no mapping.
		"""
		self.model_identifier = model_identifier
		self._layer_scoring = LayerScores(model_identifier=model_identifier, activations_model=activations_model)
		self.layers = layers
		self._logger = logging.getLogger(fullname(self))

	def __call__(self, assembly):
		return self._call(model_identifier=self.model_identifier, assembly_identifier=assembly.name, assembly=assembly)

	@store(identifier_ignore=['assembly'])
	def _call(self, model_identifier, assembly_identifier, assembly):
		benchmark = self._Benchmark(assembly)
		self._logger.debug("Finding best layer")
		layer_scores = self._layer_scoring(benchmark=benchmark, layers=self.layers)
		self._logger.debug("Layer scores (unceiled): " + ", ".join([
			f"{layer} -> {layer_scores.raw.sel(layer=layer, aggregation='center').values:.3f}"
			f"+-{layer_scores.raw.sel(layer=layer, aggregation='error').values:.3f}"
			for layer in layer_scores['layer'].values]))
		best_layer = layer_scores['layer'].values[layer_scores.sel(aggregation='center').argmax()]
		return best_layer

	class _Benchmark(BenchmarkBase):
		def __init__(self, assembly_repetition, similarity_metric=None, ceiler=None):
			assert len(np.unique(assembly_repetition['region'])) == 1
			self.region = np.unique(assembly_repetition['region'])[0]
			self.assembly = average_repetition(assembly_repetition)

			self._similarity_metric = similarity_metric or CrossRegressedCorrelation()
			identifier = f'{assembly_repetition.name}-layer_selection'
			ceiler = ceiler or InternalConsistency()
			super(LayerSelection._Benchmark, self).__init__(
				identifier=identifier, ceiling_func=lambda: ceiler(assembly_repetition))

		def __call__(self, candidate):
			candidate.start_recording(self.region)
			source_assembly = candidate.look_at(self.assembly.stimulus_set)
			raw_score = self._similarity_metric(source_assembly, self.assembly)
			return ceil_score(raw_score, self.ceiling)


class LayerScores:
	def __init__(self, model_identifier, activations_model):
		self.model_identifier = model_identifier
		self._activations_model = activations_model
		self._logger = logging.getLogger(fullname(self))

	def __call__(self, benchmark, layers, benchmark_identifier=None):
		return self._call(model_identifier=self.model_identifier,
						  benchmark_identifier=benchmark_identifier or benchmark.identifier,
						  model=self._activations_model, benchmark=benchmark, layers=layers)

	@store_xarray(identifier_ignore=['model', 'benchmark', 'layers'], combine_fields={'layers': 'layer'})
	def _call(self, model_identifier, benchmark_identifier,  # storage fields
			  model, benchmark, layers):
		# pre-run activations together to avoid running every layer separately
		model(layers=layers, stimuli=benchmark.assembly.stimulus_set)

		layer_scores = []
		for layer in tqdm(layers, desc="layers"):
			layer_model = LayerModel(identifier=model_identifier, base_model=model,
									 region_layer_map={benchmark.region: layer})
			score = benchmark(layer_model)
			score = score.expand_dims('layer')
			score['layer'] = [layer]
			layer_scores.append(score)
		layer_scores = Score.merge(*layer_scores)
		layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
		return layer_scores

class TemporalModelCommitment(BrainModel):
	def __init__(self, identifier, base_model, layers, region_layer_map: Optional[dict] = None):
		self.layers = layers
		self.identifier = identifier
		self.base_model = base_model
		self.region_layer_map = region_layer_map or {}
		self.recorded_regions = []

		self.time_bins = None
		self._temporal_maps = {}
		self._layer_regions = None

	def make_temporal(self, assembly):
		assert self.region_layer_map																	# force commit_region to come before
		assert len(set(assembly.time_bin.values)) > 1													# force temporal recordings/assembly
		
		temporal_mapped_regions = set(assembly['region'].values)
		assert bool(set(self.region_layer_map.keys()).intersection(temporal_mapped_regions))			# force simalr brain regions
		
		temporal_mapped_regions = list(set(self.region_layer_map.keys()).intersection(self.region_layer_map.keys()))
		layer_regions = {self.region_layer_map[region]: region for region in temporal_mapped_regions}
		
		stimulus_set = assembly.stimulus_set[assembly.stimulus_set['image_id'].isin(assembly['image_id'].values)]

		stimuli_identifier = self.identifier + 'temporal_map_stim'

		activations = self.base_model(stimulus_set, layers=list(layer_regions.keys()), stimuli_identifier=stimuli_identifier)
		activations = self._set_region_coords(activations, layer_regions)

		for region in temporal_mapped_regions:
			time_bin_regressor = {}
			region_activations = activations.sel(region=region)
			for time_bin in assembly.time_bin.values:
				target_assembly = assembly.sel(time_bin=time_bin,region=region)
				time_bin_regressor[time_bin] = pls_regression().fit(region_activations, target_assembly)
			self._temporal_maps[region]=time_bin_regressor

	def look_at(self, stimuli):
		layer_regions = {self.region_layer_map[region]: region for region in self.recorded_regions}
		assert len(layer_regions) == len(self.recorded_regions), f"duplicate layers for {self.recorded_regions}"
		activations = self.base_model(stimuli, layers=list(layer_regions.keys()))
		activations['region'] = 'neuroid', [layer_regions[layer] for layer in activations['layer'].values]

		activations = self._set_region_coords(activations,layer_regions)
		temporal_assembly = []
		for region in self.recorded_regions:
			temporal_regressors = self._temporal_maps[region]
			region_activations = activations.sel(region=region)
			for time_bin in self.time_bins:
				regressor = temporal_regressors[time_bin]
				regressed_act = regressor.predict(region_activations)
				regressed_act = _package_temporal(time_bin, regressed_act)
				temporal_assembly.append(regressed_act)
		temporal_assembly = merge_data_arrays(temporal_assembly)
		return temporal_assembly

	def _set_region_coords(self, activations, layer_regions):
		coords = { 'region' : ( ('neuroid'), [layer_regions[layer] for layer in activations['layer'].values]) }
		activations = activations.assign_coords(**coords)
		activations = activations.set_index({'neuroid':'region'}, append=True)
		return activations

	def _package_temporal(self, time_bin, assembly):
		assembly=assembly.expand_dims('time_bin', axis=-1)
		coords = {
					'time_bin_start': (('time_bin'), [t_bin[0]])
				  , 'time_bin_end': (('time_bin'), [t_bin[1]])
				 }
		assembly=assembly.assign_coords(**coords)
		assembly=assembly.set_index(time_bin=['time_bin_start', 'time_bin_end'])
		return assembly

	def start_temporal_recording(self, recording_target, time_bins):
		assert self._temporal_maps
		assert self.region_layer_map
		assert recording_target in self._temporal_maps.keys()
		if time_bins is not None:
			assert set(self._temporal_maps[recording_target].keys()).issuperset(set(time_bins))
		else:
			time_bins = self._temporal_maps[recording_target].keys()
		# self.layer_model.start_recording(recording_target)
		self.recorded_regions = [recording_target]
		self.time_bins = time_bins

	def start_recording(self, recording_target):
		assert self._temporal_maps
		assert self.region_layer_map
		assert recording_target in self._temporal_maps.keys()
		if self.time_bins is None:
			self.time_bins = self._temporal_maps[recording_target].keys()
		self.recorded_regions = [recording_target]

	def commit_region(self, region, assembly):
		layer_selection = LayerSelection(model_identifier=self.identifier,
										 activations_model=self.base_model, layers=self.layers)
		best_layer = layer_selection(assembly)
		self.region_layer_map[region] = best_layer

	def receptive_fields(self, record=True):
		pass


def single_element(element_list):
	assert len(element_list) == 1
	return element_list[0]
