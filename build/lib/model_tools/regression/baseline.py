import os
import hashlib
import numpy as np
import xarray as xr

from brainio_base.stimuli import StimulusSet
from brainio_base.assemblies import NeuronRecordingAssembly

from model_tools.regression import pls_regression

from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

class RegressionScore():

    def __init__(self, _model_name, _layer, _data_dir, _im_dir='images',_nc_file=None, _wrapper=None, **kwargs):
        self.data_dir = _data_dir
        self.modelname = _model_name
        self.layer = _layer
        self.identifier = self.modelname + '_' + self.layer
        self.testing_stim_dir = os.path.join(self.data_dir, _im_dir)

        self.assembly = load_nc_data(self.data_dir, _nc_file)
        self.extractor = _wrapper

        # self.stimuli_identifier_train = self.identifier + '_stim_train'
        # self.stimuli_identifier_validation = self.identifier + '_stim_validation'
        self.stimuli_identifier_train = None
        self.stimuli_identifier_validation = None

    def __call__(self, im_path_offset=0):
        # split test/train
        train_index, test_index, resp_train, resp_val = self.split_data()
        regressor = pls_regression()
        train_stim_set = self.create_stim_set(train_index + im_path_offset)
        validation_stim_set = self.create_stim_set(test_index + im_path_offset)
        # Testing activations:
        test_activations = self.extractor(train_stim_set, layers=[self.layer], stimuli_identifier=self.stimuli_identifier_train)
        # Scale
        resp_train.values = scale(resp_train.values)
        # fit
        regressor.fit(test_activations, resp_train)
        # Validation activations:
        validation_activations = self.extractor(validation_stim_set, layers=[self.layer], stimuli_identifier=self.stimuli_identifier_validation)
        # Predict
        predicted = regressor.predict(validation_activations)
        # Scale
        resp_val.values = scale(resp_val.values)
        # Score
        return np.sqrt(mean_squared_error(predicted, resp_val.values))

    def split_data(self, test_size=0.20):
        image_index = np.array([i for i in range(self.assembly.shape[0])])
        im_train, im_val, resp_train, resp_val = train_test_split(image_index, self.assembly, test_size=test_size, random_state=123)
        return im_train, im_val, resp_train, resp_val

    def create_stim_set(self, im_idx):
        stim_paths = [self.testing_stim_dir + '/image_{:05}.jpg'.format(i) for i in im_idx]
        basenames = [os.path.splitext(os.path.basename(path))[0] for path in stim_paths]
        image_ids = [get_im_hash(im_path) for im_path in stim_paths]
        s = StimulusSet({'image_file_path': stim_paths, 'image_file_name': basenames, 'image_id': image_ids})
        s.image_paths = {image_id: path for image_id, path in zip(image_ids, stim_paths)}
        s.name = None
        return s

    def generate_stim_paths(self, im_idx):
        stim_paths = [self.testing_stim_dir + '/image_{:05}.jpg'.format(i) for i in im_idx]
        return stim_paths

def load_nc_data(_data_dir, fname):
    # nc_load = xr.open_dataarray(os.path.join(_data_dir, 'nc_files' ,fname))
    nc_load = xr.open_dataarray(os.path.join(_data_dir, fname))
    assembly = NeuronRecordingAssembly(nc_load)
    assembly = assembly.squeeze("time_bin")
    assembly = assembly.transpose('presentation', 'neuroid')
    # fill nan with response mean
    assembly = assembly.fillna(assembly.mean(dim=('neuroid'), skipna=True))
    return assembly

def get_im_hash(path):
    buffer_size = 64 * 2 ** 10
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        buffer = f.read(buffer_size)
        while len(buffer) > 0:
            sha1.update(buffer)
            buffer = f.read(buffer_size)
    return sha1.hexdigest()
    