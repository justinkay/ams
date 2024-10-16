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

        all_preds = dataset.preds
        H, N, C = all_preds.shape

        if loss_fn is not None:
            self.true_losses = []
            for h in range(H):
                # TODO: will need to change this when we introduce other loss functions
                ce_loss = loss_fn(all_preds[h], torch.tensor(self.labels), reduction='none')
                self.true_losses.append(ce_loss.mean())
        
        if accuracy_fn is not None:
            self.true_accs = []
            for h in range(H):
                self.true_accs.append(accuracy_fn(torch.softmax(all_preds[h], dim=-1), torch.tensor(self.labels)))

        self.labels = torch.tensor(self.labels, device=dataset.device)

    def __call__(self, idx):
        return self.labels[idx].item()

class User:
    # not implemented
    pass
