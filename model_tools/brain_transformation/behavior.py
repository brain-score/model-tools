import os
from collections import OrderedDict

import sklearn.linear_model
import sklearn.multioutput

from brainio_base.assemblies import walk_coords, array_is_element, BehavioralAssembly
from brainscore.model_interface import BrainModel
from model_tools.brain_transformation.behavior_classifier import TFProbabilitiesClassifier


class BehaviorArbiter(BrainModel):
    def __init__(self, mapping):
        self.mapping = mapping
        self.current_executor = None

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        self.current_executor = self.mapping[task]
        return self.current_executor.start_task(task, *args, **kwargs)

    def look_at(self, stimuli, *args, **kwargs):
        return self.current_executor.look_at(stimuli, *args, **kwargs)


class LogitsBehavior(BrainModel):
    def __init__(self, identifier, activations_model):
        self.identifier = identifier
        self.activations_model = activations_model
        self.current_task = None

    def start_task(self, task: BrainModel.Task, fitting_stimuli):
        assert task in [BrainModel.Task.passive, BrainModel.Task.label]
        if task == BrainModel.Task.label:
            assert fitting_stimuli == 'imagenet'
        self.current_task = task

    def look_at(self, stimuli):
        if self.current_task is BrainModel.Task.passive:
            return
        logits = self.activations_model(stimuli, layers=['logits'])
        assert len(logits['neuroid']) == 1000
        logits = logits.transpose('presentation', 'neuroid')
        prediction_indices = logits.values.argmax(axis=1)
        with open(os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt')) as f:
            synsets = f.read().splitlines()
        prediction_synsets = [synsets[index] for index in prediction_indices]
        return prediction_synsets


class ProbabilitiesMapping(BrainModel):
    def __init__(self, identifier, activations_model, layer):
        self.identifier = identifier
        self.activations_model = activations_model
        self.layer = layer
        self.classifier = TFProbabilitiesClassifier()
        self.current_task = None

    def start_task(self, task: BrainModel.Task, fitting_stimuli):
        assert task in [BrainModel.Task.passive, BrainModel.Task.probabilities]
        self.current_task = task

        fitting_features = self.activations_model(fitting_stimuli, layers=[self.layer])
        fitting_features = fitting_features.transpose('presentation', 'neuroid')
        assert all(fitting_features['image_id'].values == fitting_stimuli['image_id'].values), \
            "image_id ordering is incorrect"
        self.classifier.fit(fitting_features, fitting_stimuli['image_label'])

    def look_at(self, stimuli):
        if self.current_task is BrainModel.Task.passive:
            return
        features = self.activations_model(stimuli, layers=[self.layer])
        features = features.transpose('presentation', 'neuroid')
        prediction = self.classifier.predict_proba(features)
        return prediction
