import csv

from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers

from model_tools.activations.pca import LayerPCA
from model_tools.brain_transformation import TemporalModelCommitment

from brainscore import get_assembly
from brainscore.metrics.regression import CrossRegressedCorrelation
from brainscore.benchmarks.loaders import AssemblyLoader, DicarloMajaj2015Loader, DicarloMajaj2015ITLoader

from result_caching import store

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

train_test_assembly_loader = DicarloMajaj2015TemporalLoader()
commit_loader = DicarloMajaj2015ITLoader()

training_assembly = train_test_assembly_loader(variation=3)     # temporal, variation 3 regression training data
commit_assembly = commit_loader(average_repetition=False)       # time-averaged, variation 6 neurorecordings
validation_assembly = train_test_assembly_loader(variation=6)   # temporal, variation 6 validation data

brain_region = 'IT'

model_name = 'vgg-19'
extractor = base_model_pool[model_name]
extractor._ensure_loaded()
LayerPCA.hook(extractor, n_components=1000)
layers = model_layers[model_name]

t_bins = [t for t in training_assembly.time_bin.values if t[0] >= 0]

temporal_model = TemporalModelCommitment('vgg-19-temporal', extractor, layers)
temporal_model.commit_region(brain_region, commit_assembly)                     # best_layer for brain region
temporal_model.make_temporal(training_assembly)
temporal_model.start_temporal_recording(brain_region, t_bins)

# Validation:
stim = get_stim(commit_assembly)
temporal_activations = temporal_model.look_at(stim)

@store(identifier_ignore=['time_bins','activations','assembly'])
def _get_temporal_scores(identifier, region, time_bins, activations, assembly):
    temp_scores = []
    for t in time_bins:
        scorer = CrossRegressedCorrelation()
        source_assembly = activations.sel(time_bin=t)
        target_assembly = assembly.sel(time_bin=t, region=region)
        score = scorer(source_assembly, target_assembly).values
        temp_scores.append([t[0], t[1], score[0], score[1]])
    return temp_scores

temporal_scores=_get_temporal_scores('vgg-19_IT_temporal', brain_region, t_bins, temporal_activations, validation_assembly)

import os

save_path = os.path.expanduser('~')
csv_file = os.path.join(save_path, 'temporal_scores.csv')

with open(csv_file, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['time_bin_start', 'time_bin_end', 'score', 'error'])
    writer.writerows(temporal_scores)