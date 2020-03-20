import cv2
import logging
import numpy as np
from tqdm import tqdm

from brainscore.model_interface import BrainModel
from brainscore.utils import fullname


class VisualSearchObjArray(BrainModel):
    def __init__(self, identifier, target_model_param, stimuli_model_param):
        self.current_task = None
        self.identifier = identifier
        self.target_model = target_model_param['target_model']
        self.stimuli_model = stimuli_model_param['stimuli_model']
        self.target_layer = target_model_param['target_layer']
        self.stimuli_layer = stimuli_model_param['stimuli_layer']
        self.search_image_size = stimuli_model_param['search_image_size']
        self._logger = logging.getLogger(fullname(self))

    def start_task(self, task: BrainModel.Task, **kwargs):
        self.fix = kwargs['fix']  # fixation map
        self.max_fix = kwargs['max_fix']  # maximum allowed fixation excluding the very first fixation
        self.data_len = kwargs['data_len']  # Number of stimuli
        self.current_task = task

    def look_at(self, stimuli_set):
        self.gt_array = []
        gt = stimuli_set[stimuli_set['image_label'] == 'mask']
        gt_paths = list(gt.image_paths.values())[int(gt.index.values[0]):int(gt.index.values[-1] + 1)]

        for i in range(6):
            imagename_gt = gt_paths[i]

            gt = cv2.imread(imagename_gt, 0)
            gt = cv2.resize(gt, (self.search_image_size, self.search_image_size), interpolation=cv2.INTER_AREA)
            retval, gt = cv2.threshold(gt, 125, 255, cv2.THRESH_BINARY)
            temp_stim = np.uint8(np.zeros((3 * self.search_image_size, 3 * self.search_image_size)))
            temp_stim[self.search_image_size:2 * self.search_image_size,
            self.search_image_size:2 * self.search_image_size] = np.copy(gt)
            gt = np.copy(temp_stim)
            gt = gt / 255

            self.gt_array.append(gt)

        self.gt_total = np.copy(self.gt_array[0])
        for i in range(1, 6):
            self.gt_total += self.gt_array[i]

        self.score = np.zeros((self.data_len, self.max_fix + 1))
        self.data = np.zeros((self.data_len, self.max_fix + 2, 2), dtype=int)
        S_data = np.zeros((300, 7, 2), dtype=int)
        I_data = np.zeros((300, 1), dtype=int)

        data_cnt = 0

        target = stimuli_set[stimuli_set['image_label'] == 'target']
        target_features = self.target_model(target, layers=[self.target_layer], stimuli_identifier=False)
        if target_features.shape[0] == target_features['neuroid_num'].shape[0]:
            target_features = target_features.T

        stimuli = stimuli_set[stimuli_set['image_label'] == 'stimuli']
        stimuli_features = self.stimuli_model(stimuli, layers=[self.stimuli_layer], stimuli_identifier=False)
        if stimuli_features.shape[0] == stimuli_features['neuroid_num'].shape[0]:
            stimuli_features = stimuli_features.T

        import torch

        for i in tqdm(range(self.data_len), desc="visual search stimuli: "):
            op_target = self.unflat(target_features[i:i + 1])
            MMconv = torch.nn.Conv2d(op_target.shape[1], 1, kernel_size=(op_target.shape[2], op_target.shape[3]),
                                     stride=1, bias=False)
            MMconv.weight = torch.nn.Parameter(torch.Tensor(op_target))

            gt_idx = target_features.tar_obj_pos.values[i]
            gt = self.gt_array[gt_idx]

            op_stimuli = self.unflat(stimuli_features[i:i + 1])
            out = MMconv(torch.Tensor(op_stimuli)).detach().numpy()
            out = out.reshape(out.shape[2:])

            out = out - np.min(out)
            out = out / np.max(out)
            out *= 255
            out = np.uint8(out)
            out = cv2.resize(out, (self.search_image_size, self.search_image_size), interpolation=cv2.INTER_AREA)
            out = cv2.GaussianBlur(out, (7, 7), 3)

            temp_stim = np.uint8(np.zeros((3 * self.search_image_size, 3 * self.search_image_size)))
            temp_stim[self.search_image_size:2 * self.search_image_size,
            self.search_image_size:2 * self.search_image_size] = np.copy(out)
            attn = np.copy(temp_stim * self.gt_total)

            saccade = []
            (x, y) = int(attn.shape[0] / 2), int(attn.shape[1] / 2)
            saccade.append((x, y))

            for k in range(self.max_fix):
                (x, y) = np.unravel_index(np.argmax(attn), attn.shape)

                fxn_x, fxn_y = x, y

                fxn_x, fxn_y = max(fxn_x, self.search_image_size), max(fxn_y, self.search_image_size)
                fxn_x, fxn_y = min(fxn_x, (attn.shape[0] - self.search_image_size)), min(fxn_y, (
                            attn.shape[1] - self.search_image_size))

                saccade.append((fxn_x, fxn_y))

                attn, t = self.remove_attn(attn, saccade[-1][0], saccade[-1][1])

                if (t == gt_idx):
                    self.score[data_cnt, k + 1] = 1
                    data_cnt += 1
                    break

            saccade = np.asarray(saccade)
            j = saccade.shape[0]

            for k in range(j):
                tar_id = self.get_pos(saccade[k, 0], saccade[k, 1], 0)
                saccade[k, 0] = self.fix[tar_id][0]
                saccade[k, 1] = self.fix[tar_id][1]

            I_data[i, 0] = min(7, j)
            S_data[i, :j, 0] = saccade[:, 0].reshape((-1,))[:7]
            S_data[i, :j, 1] = saccade[:, 1].reshape((-1,))[:7]

            self.data[:, :7, :] = S_data
            self.data[:, 7, :] = I_data

        return (self.score, self.data)

    def remove_attn(self, img, x, y):
        t = -1
        for i in range(5, -1, -1):
            fxt_place = self.gt_array[i][x, y]
            if (fxt_place > 0):
                t = i
                break

        if (t > -1):
            img[self.gt_array[t] == 1] = 0

        return img, t

    def get_pos(self, x, y, t):
        for i in range(5, -1, -1):
            fxt_place = self.gt_array[i][int(x), int(y)]
            if (fxt_place > 0):
                t = i + 1
                break
        return t

    def unflat(self, X):
        channel_names = ['channel', 'channel_x', 'channel_y']
        assert all(hasattr(X, coord) for coord in channel_names)
        shapes = [len(set(X[channel].values)) for channel in channel_names]
        X = np.reshape(X.values, [X.shape[0]] + shapes)
        X = np.transpose(X, axes=[0, 3, 1, 2])
        return X
