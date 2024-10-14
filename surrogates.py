
import torch
import numpy as np


class Ensemble:

    def __init__(self, all_preds):
        self.H, self.N, self.C = all_preds.shape # H=num_hypothesis, N=num_data_points, C=num_classes
        self.all_preds = all_preds

        # indices of all models in ensemble; at first, all are included; they may be pruned
        self.ensemble_idxs = list(range(self.H))

    def get_preds(self):
        surrogate_preds = np.mean(self.all_preds[self.ensemble_idxs, :, :], axis=0)  # Shape: (N, C)
        return torch.tensor(surrogate_preds)