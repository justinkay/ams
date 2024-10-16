from datasets import DomainNet126
import numpy as np
import torch
from scipy.special import softmax


class Oracle:
    
    def __init__(self, dataset, base_dir="../powerful-benchmarker/datasets/",
                 loss_fn=None, accuracy_fn=None):
        if isinstance(dataset, DomainNet126):
            test_set = dataset.task.split("_")[-1]
            if dataset.use_target_val:
                test_set_txt = f'{base_dir}/domainnet/{test_set}126_test.txt'
            else:
                test_set_txt = f'{base_dir}/domainnet/{test_set}126_train.txt'
        else:
            raise NotImplementedError()
        
        with open(test_set_txt, 'r') as f:
            self.labels = np.array([ int(s.split(" ")[-1].replace("\n","")) for s in f.readlines()])
            self.labels = torch.tensor(self.labels, device=dataset.device)

        all_preds = dataset.preds
        H, N, C = all_preds.shape

        if loss_fn is not None:
            print("Computing losses...")
            self.true_losses = []
            # Compute losses for all models at once
            ce_losses = loss_fn(all_preds.reshape(-1, C), self.labels.repeat(H), reduction='none')
            self.true_losses = ce_losses.reshape(H, -1).mean(dim=1).tolist()
            print("Losses computed")
            
        if accuracy_fn is not None:
            print("Computing accuracies..."
            self.true_accs = []
            # Compute accuracies for all models at once
            softmax_preds = torch.softmax(all_preds, dim=-1)
            self.true_accs = accuracy_fn(softmax_preds.reshape(-1, C), self.labels.repeat(H)).reshape(H).tolist()
            print("Accuracies computed")

    def __call__(self, idx):
        return self.labels[idx].item()

class User:
    # not implemented
    pass
