import os
from collections import OrderedDict

from brainio_base.assemblies import BehavioralAssembly
from brainscore.model_interface import BrainModel

import cv2
import numpy as np
from tqdm import tqdm
import torch

class VisualSearchObjArray(BrainModel):
    def __init__(self, target_model, target_layer, target_model_winsize, stimulus_model, stimulus_layer):
        self.target_model = target_model
        self.target_layer = target_layer
        self.target_model_winsize = target_model_winsize
        self.stimulus_model = stimulus_model
        self.stimulus_layer = stimulus_layer
        self.current_task = None
        self.eye_res = 224
        self.arr_size = 6
        self.data_len = 300

    def start_task(self, task: BrainModel.Task):
        self.current_task = task

    def look_at(self, stimuli_set):
        self.gt_array = []
        gt = stimuli_set[stimuli_set['image_label'] == 'gt']
        gt_paths = list(gt.image_paths.values())[int(gt.index.values[0]):int(gt.index.values[-1]+1)]

        for i in range(6):
            imagename_gt = gt_paths[i]

            gt = cv2.imread(imagename_gt, 0)
            gt = cv2.resize(gt, (self.eye_res, self.eye_res), interpolation = cv2.INTER_AREA)
            retval, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
            temp_stim = np.uint8(np.zeros((3*self.eye_res, 3*self.eye_res)))
            temp_stim[self.eye_res:2*self.eye_res, self.eye_res:2*self.eye_res] = np.copy(gt)
            gt = np.copy(temp_stim)
            gt = gt/255

            self.gt_array.append(gt)

        self.gt_total = np.copy(self.gt_array[0])
        for i in range(1,6):
            self.gt_total += self.gt_array[i]

        self.score = np.zeros((self.data_len, self.arr_size+1))
        self.data = np.zeros((self.data_len, self.arr_size+1, 2), dtype=int)
        data_cnt = 0

        target = stimuli_set[stimuli_set['image_label'] == 'target']
        target_features = self.target_model(target, layers=[self.target_layer])
        # target_paths = list(target.image_paths.values())[int(target.index.values[0]):int(target.index.values[-1]+1)]

        stimuli = stimuli_set[stimuli_set['image_label'] == 'stimuli']
        stimuli_features = self.stimuli_model(stimuli, layers=[self.stimuli_layer])
        # stimuli_paths = list(stimuli.image_paths.values())[int(stimuli.index.values[0]):int(stimuli.index.values[-1]+1)]

        for i in tqdm(range(self.data_len)):
            op_target = target_features[i]
            MMconv = torch.nn.Conv2d(self.target_model_winsize, 1, kernel_size=1, stride=1, bias=False)
            MMconv.weight = torch.nn.Parameter(op_target)

            gt_idx = stimuli.tar_obj_pos.values[i]
            gt = self.gt_array[gt_idx]

            op_stimuli = stimuli_features[i]
            out = MMconv(op_stimuli).cpu().detach().numpy()
            out = out.reshape(out.shape[2:])

            out = out - np.min(out)
            out = out/np.max(out)
            out *= 255
            out = np.uint8(out)
            out = cv2.resize(out, (eye_res, eye_res), interpolation = cv2.INTER_AREA)
            out = cv2.GaussianBlur(out,(7,7),3)

            temp_stim = np.uint8(np.zeros((3*self.eye_res, 3*self.eye_res)))
            temp_stim[self.eye_res:2*self.eye_res, self.eye_res:2*self.eye_res] = np.copy(out)
            attn = np.copy(temp_stim*self.gt_total)

            saccade = []
            (x, y) = int(attn.shape[0]/2), int(attn.shape[1]/2)
            saccade.append((x, y))

            j = 1

            for k in range(self.arr_size):
                (x, y) = np.unravel_index(np.argmax(attn), attn.shape)

                fxn_x, fxn_y = x, y

                fxn_x, fxn_y = max(fxn_x, eye_res), max(fxn_y, eye_res)
                fxn_x, fxn_y = min(fxn_x, (attn.shape[0]-eye_res)), min(fxn_y, (attn.shape[1]-eye_res))

                saccade.append((fxn_x, fxn_y))

                attn, t = VisualSearchObjArray.remove_attn(attn, saccade[-1][0], saccade[-1][1])

                if(t==gt_idx):
                    score[data_cnt, k+1] = 1
                    j = k+2
                    data_cnt += 1
                    break

            saccade = np.asarray(saccade)
            self.data[i, :j, 0] = saccade[:, 0].reshape((-1,))
            self.data[i, :j, 1] = saccade[:, 1].reshape((-1,))

        return (self.score, data)

    def remove_attn(self, img, x, y):
        t = -1
        for i in range(5, -1, -1):
            fxt_place = gt_array[i][x, y]
            if (fxt_place>0):
                t = i
                break

        if(t>-1):
            img[gt_array[t] == 1] = 0

        return img, t
