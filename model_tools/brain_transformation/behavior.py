from collections import OrderedDict

import sklearn.linear_model
import sklearn.multioutput

from brainio_base.assemblies import walk_coords, array_is_element, BehavioralAssembly
from brainscore.model_interface import BrainModel
from model_tools.utils import make_list


class BehaviorArbiter(BrainModel):
    def __init__(self, mapping):
        self.mapping = mapping
        self.current_executor = None

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        self.current_executor = self.mapping[task]
        return self.current_executor.start_task(task, *args, **kwargs)

    def look_at(self, stimuli, *args, **kwargs):
        return self.current_executor.look_at(stimuli, *args, **kwargs)


class LabelBehavior(BrainModel):
    def __init__(self, probabilities_mapping):
        self.probabilities_mapping = probabilities_mapping

    def start_task(self, task: BrainModel.Task, fitting_stimuli):
        assert task in [BrainModel.Task.passive, BrainModel.Task.label]
        if task == BrainModel.Task.label:
            self.probabilities_mapping.start_task(BrainModel.Task.probabilities, fitting_stimuli=fitting_stimuli)

    def look_at(self, stimuli, number_of_trials=1):
        probabilities = self.probabilities_mapping.look_at(stimuli=stimuli, number_of_trials=number_of_trials)
        assert len(probabilities['choice']) == 1_000
        prediction_indices = probabilities.argmax('choice').values
        predictions = probabilities['choice'].values[prediction_indices]
        return BehavioralAssembly(predictions, coords={
            **{coord: (dims, values) for coord, dims, values in walk_coords(probabilities['presentation'])},
            **{'synset': ('presentation', predictions),
               'logit': ('presentation', predictions),
               'label': ('presentation', predictions),
               'choice': ('presentation', predictions)}},
                                  dims=['presentation'])


class ProbabilitiesMapping(BrainModel):
    def __init__(self, identifier, activations_model, layer):
        """
        :param identifier: a string to identify the model
        :param activations_model: the model from which to retrieve representations for stimuli
        :param layer: the single behavioral readout layer or a list of layers to read out of.
        """
        self.identifier = identifier
        self.activations_model = activations_model
        self.readout = make_list(layer)
        self.classifier = ProbabilitiesMapping.ProbabilitiesClassifier()
        self.current_task = None

    def start_task(self, task: BrainModel.Task, fitting_stimuli):
        assert task in [BrainModel.Task.passive, BrainModel.Task.probabilities]
        self.current_task = task

        fitting_features = self.activations_model(fitting_stimuli, layers=self.readout)
        fitting_features = fitting_features.transpose('presentation', 'neuroid')
        assert all(fitting_features['image_id'].values == fitting_stimuli['image_id'].values), \
            "image_id ordering is incorrect"
        self.classifier.fit(fitting_features, fitting_stimuli['image_label'])

    def look_at(self, stimuli, number_of_trials=1):
        if self.current_task is BrainModel.Task.passive:
            return
        features = self.activations_model(stimuli, layers=self.readout)
        features = features.transpose('presentation', 'neuroid')
        prediction = self.classifier.predict_proba(features)
        return prediction

    class ProbabilitiesClassifier:
        def __init__(self, classifier_c=1e-3):
            self._classifier = sklearn.linear_model.LogisticRegression(
                multi_class='multinomial', solver='newton-cg', C=classifier_c)
            self._label_mapping = None
            self._scaler = None

        def fit(self, X, Y):
            self._scaler = sklearn.preprocessing.StandardScaler().fit(X)
            X = self._scaler.transform(X)
            Y, self._label_mapping = self.labels_to_indices(Y.values)
            self._classifier.fit(X, Y)
            return self

        def predict_proba(self, X):
            assert len(X.shape) == 2, "expected 2-dimensional input"
            scaled_X = self._scaler.transform(X)
            proba = self._classifier.predict_proba(scaled_X)
            # we take only the 0th dimension because the 1st dimension is just the features
            X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                        if array_is_element(dims, X.dims[0])}
            proba = BehavioralAssembly(proba,
                                       coords={**X_coords, **{'choice': list(self._label_mapping.values())}},
                                       dims=[X.dims[0], 'choice'])
            return proba

        def labels_to_indices(self, labels):
            label2index = OrderedDict()
            indices = []
            for label in labels:
                if label not in label2index:
                    label2index[label] = (max(label2index.values()) + 1) if len(label2index) > 0 else 0
                indices.append(label2index[label])
            index2label = OrderedDict((index, label) for label, index in label2index.items())
            return indices, index2label
