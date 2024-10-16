import torch
import numpy as np


class Ensemble:

    def __init__(self, all_preds):
        self.all_preds = all_preds
        self.device = all_preds.device
        H, N, C = all_preds.shape
        self.ensemble_idxs = torch.arange(H, device=self.device)

    def get_preds(self):
        return self.all_preds[self.ensemble_idxs].mean(dim=0)
