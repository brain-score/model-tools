from typing import Optional

from brainio_base.assemblies import merge_data_arrays

from model_tools.brain_transformation import LayerSelection

from brainscore.model_interface import BrainModel
from brainscore.metrics.regression import pls_regression

from result_caching import store, store_dict

class TemporalModelCommitment(BrainModel):
    def __init__(self, identifier, base_model, layers, region_layer_map: Optional[dict] = None):
        self.layers = layers
        self.identifier = identifier or None
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
        assert bool \
            (set(self.region_layer_map.keys()).intersection(temporal_mapped_regions))			        # force simalr brain regions

        temporal_mapped_regions = list(set(self.region_layer_map.keys()).intersection(self.region_layer_map.keys()))
        layer_regions = {self.region_layer_map[region]: region for region in temporal_mapped_regions}

        stimulus_set = assembly.stimulus_set[assembly.stimulus_set['image_id'].isin(assembly['image_id'].values)]

        stimuli_identifier = None
        if self.identifier is not None:
            stimuli_identifier = self.identifier + 'temporal_map_stim'

        activations = self.base_model(stimulus_set, layers=list(layer_regions.keys()), stimuli_identifier=stimuli_identifier)
        activations = self._set_region_coords(activations, layer_regions)

        self._temporal_maps = self._set_temporal_maps(self.identifier, temporal_mapped_regions, activations, assembly)

    def look_at(self, stimuli):
        layer_regions = {self.region_layer_map[region]: region for region in self.recorded_regions}
        assert len(layer_regions) == len(self.recorded_regions), f"duplicate layers for {self.recorded_regions}"
        activations = self.base_model(stimuli, layers=list(layer_regions.keys()))

        activations = self._set_region_coords(activations ,layer_regions)
        return self._temporal_activations(self.identifier, activations)

    @store(identifier_ignore=['assembly'])
    def _temporal_activations(self, identifier, assembly):
        temporal_assembly = []
        for region in self.recorded_regions:
            temporal_regressors = self._temporal_maps[region]
            region_activations = assembly.sel(region=region)
            for time_bin in self.time_bins:
                regressor = temporal_regressors[time_bin]
                regressed_act = regressor.predict(region_activations)
                regressed_act = self._package_temporal(time_bin, region, regressed_act)
                temporal_assembly.append(regressed_act)
        temporal_assembly = merge_data_arrays(temporal_assembly)
        return temporal_assembly

    @store_dict(dict_key='temporal_mapped_regions', identifier_ignore=['temporal_mapped_regions', 'activations' ,'assembly'])
    def _set_temporal_maps(self, identifier, temporal_mapped_regions, activations, assembly):
        temporal_maps = {}
        for region in temporal_mapped_regions:
            time_bin_regressor = {}
            region_activations = activations.sel(region=region)
            for time_bin in assembly.time_bin.values:
                target_assembly = assembly.sel(time_bin=time_bin ,region=region)
                regressor = pls_regression(neuroid_coord=('neuroid_id' ,'layer' ,'region'))
                regressor.fit(region_activations, target_assembly)
                time_bin_regressor[time_bin] = regressor
            temporal_maps[region] = time_bin_regressor
        return temporal_maps

    def _set_region_coords(self, activations, layer_regions):
        coords = { 'region' : ( ('neuroid'), [layer_regions[layer] for layer in activations['layer'].values]) }
        activations = activations.assign_coords(**coords)
        activations = activations.set_index({'neuroid' :'region'}, append=True)
        return activations

    def _package_temporal(self, time_bin, region, assembly):
        assert len(time_bin) == 2
        assembly = assembly.expand_dims('time_bin', axis=-1)
        coords = {
            'time_bin_start': (('time_bin'), [time_bin[0]])
            , 'time_bin_end': (('time_bin'), [time_bin[1]])
            , 'region' : ( ('neuroid'), [region] * assembly.shape[1])
        }
        assembly = assembly.assign_coords(**coords)
        assembly = assembly.set_index(time_bin=['time_bin_start', 'time_bin_end'], neuroid='region', append=True)
        return assembly

    def start_temporal_recording(self, recording_target, time_bins):
        assert self._temporal_maps
        assert self.region_layer_map
        assert recording_target in self._temporal_maps.keys()
        if time_bins is not None:
            assert set(self._temporal_maps[recording_target].keys()).issuperset(set(time_bins))
        else:
            time_bins = self._temporal_maps[recording_target].keys()

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