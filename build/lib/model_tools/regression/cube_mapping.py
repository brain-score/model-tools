import os
from numpy import linalg as la
from numpy import sqrt, concatenate
from model_tools.regression.mapper import Mapper
from model_tools.regression.baseline import RegressionScore
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error

class CubeMapper(RegressionScore):
    def __init__(self, *args, **kwargs):
        super(CubeMapper, self).__init__(*args, **kwargs)
        #
        self.batch_data = BatchHook(self.extractor)
        self.cuber = None

    def init_mapper(self, num_neurons, batch_size=50, lr=1e-2, ls=0.05, ld=0.1, tol=1e-2, max_epochs=10
                    , map_type='separable', inits=None, log_rate=100, decay_rate=200, gpu_options=None, **kwargs):

        self.cuber = Mapper(num_neurons=num_neurons, batch_size=batch_size, init_lr=lr, ls=ls, ld=ld, tol=tol
                            , max_epochs=max_epochs, map_type=map_type, inits=inits, log_rate=log_rate
                            , decay_rate=decay_rate, gpu_options=gpu_options, **kwargs)

    def __call__(self, im_path_offset=0):
        # split test/train
        train_index, test_index, resp_train, resp_val = self.split_data()

        train_stim_set = self.create_stim_set(train_index + im_path_offset)
        validation_stim_set = self.create_stim_set(test_index + im_path_offset)

        # Testing activations:
        self.extractor(train_stim_set, layers=[self.layer])
        # Normalize layer outputs
        _batchArray = norm_batch(self.batch_data.batchArray)
        #
        train_gt = scale(resp_train.values)
        self.fit(_batchArray, train_gt)

        self.batch_data.reset()

        # Validation activations:
        self.extractor(validation_stim_set, layers=[self.layer])
        # Normalize layer outputs
        _batchArray = norm_batch(self.batch_data.batchArray)

        # Score
        predicted = self.predict(_batchArray)
        #
        val_gt = scale(resp_val.values)
        #
        return sqrt(mean_squared_error(predicted, val_gt))

    def save_weights(self, save_dir, fname):
        self.cuber.save_weights(os.path.join(save_dir, fname))

    def predict(self, data):
        return self.cuber.predict(data)

    def fit(self, data_test, data_gt):
        self.cuber.fit(data_test, data_gt)


def norm_batch(data):
    return data/la.norm(data, axis=-1, keepdims=True, ord=1)

class BatchHook(object):
    def __init__(self, extractor):
        self.batchArray = None
        self.extractor = extractor
        self.hook()

    def reset(self):
        self.batchArray = None

    def __call__(self, batch_activations):
        activation = list(batch_activations.values())[0]
        if self.batchArray is None:
            self.batchArray = activation
        else:
            self.batchArray = concatenate( (self.batchArray, activation) )
        return batch_activations

    def hook(self):
        hook = self
        handle = self.extractor.register_batch_hook(hook)
        hook.handle = handle
        return handle
        