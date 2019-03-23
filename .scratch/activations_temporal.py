from candidate_models.base_models import base_model_pool
from candidate_models.model_commitments import model_layers

from brainscore.benchmarks.loaders import DicarloMajaj2015TemporalLoader
from brainscore.metrics.regression import CrossRegressedCorrelation
from brainscore.metrics.transformations import subset

from model_tools.activations.pca import LayerPCA

import csv
import numpy as np
# Tensorflow GPU config:
import tensorflow as tf
from tensorflow import GPUOptions

from keras.backend.tensorflow_backend import set_session

gpu_config = GPUOptions()
gpu_config.allow_growth = True
gpu_config.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)))
loader = DicarloMajaj2015TemporalLoader()
assembly = loader().sel(region="IT")
# get proper time bins:
t_bins = [t for t in assembly.time_bin.values if t[0] >= 0]
# get stimulus:
stim_set = assembly.stimulus_set[
            assembly.stimulus_set['image_id'].isin(assembly['image_id'].values)]
# testing model:
model_name = 'vgg-19'
extractor = base_model_pool[model_name]
extractor._ensure_loaded()
LayerPCA.hook(extractor,n_components=1000)
# capture all layers:
layers = model_layers[model_name][-3]
# for testing:
if isinstance(layers, str):
    layers = [layers]
else:
    layers = layers[-6:-3]
test_activations = extractor(stim_set, layers)
# test_activations['region'] = 'neuroid', [layer for layer in test_activations['layer'].values]
# test_activations.expand_dims('time_bin', axis=-1)
print(layers)
#
# out_put = []
# for t in t_bins:
#     print('calculating time bin: {}'.format(t))
#     for layer in layers:
#         print('processing layer: {}'.format(layer))
#         cross_score = CrossRegressedCorrelation()
#         layer_act = test_activations.where(test_activations.layer==layer)
#         layer_act = layer_act.dropna('neuroid')
#         ass_act_t = assembly.sel(time_bin=t)
#         score = cross_score(layer_act, ass_act_t).values
#         out_put.append( [layer, t, score[0], score[1]] )
#
# with open('vgg-19-temporal_layer_scores.csv', 'w') as file:
#     writer = csv.writer(file)
#     writer.writerow(['layer','time_bin','score', 'error'])
#     writer.writerows(out_put)




