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

        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.device = dataset.device

        if loss_fn is not None:
            self.true_losses = self.compute_losses(dataset.preds)
        
        if accuracy_fn is not None:
            self.true_accs = self.compute_accuracies(dataset.preds)

    def compute_losses(self, preds):
        H, N, C = preds.shape
        losses = []
        batch_size = 100  # Adjust this value based on your GPU memory
        for i in range(0, H, batch_size):
            batch_end = min(i + batch_size, H)
            batch_preds = preds[i:batch_end]
            batch_labels = self.labels 
            batch_losses = self.loss_fn(batch_preds.view(-1, C), batch_labels.repeat(batch_end - i), reduction='none')
            batch_losses = batch_losses.view(batch_end - i, N).mean(dim=1) # Compute mean loss for each model
            losses.extend(batch_losses.tolist())
            torch.cuda.empty_cache()

        return losses

    def compute_accuracies(self, preds):
        H, N, C = preds.shape
        accuracies = []
        batch_size = 100  # Adjust this value based on your GPU memory
        for i in range(0, H, batch_size):
            batch_end = min(i + batch_size, H)
            batch_preds = torch.softmax(preds[i:batch_end], dim=-1)
            batch_labels = self.labels
            batch_acc = []
            for j in range(batch_end - i):
                acc = self.accuracy_fn(batch_preds[j], batch_labels)
                batch_acc.append(acc)
            accuracies.extend([acc.item() for acc in batch_acc])
            torch.cuda.empty_cache()
        return accuracies

    def __call__(self, idx):
        return self.labels[idx].item()

class User:
    # not implemented
    pass
