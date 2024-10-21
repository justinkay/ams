import torch


class Ensemble:
    def __init__(self, pred_logits):
        self.pred_logits = pred_logits
        self.device = pred_logits.device
        H, N, C = pred_logits.shape

    def get_preds(self, **kwargs):
        return torch.softmax(self.pred_logits, dim=-1).mean(dim=0)

class WeightedEnsemble(Ensemble):
    def __init__(self, pred_logits):
        super().__init__(pred_logits)

    def get_preds(self, weights):
        return (torch.softmax(self.pred_logits, dim=-1) * weights.view(-1, 1, 1)).sum(dim=0)
    
class OracleSurrogate:
    def __init__(self, oracle):
        self.oracle = oracle
        self.device = oracle.device
        self.pred_logits = self.oracle.dataset.pred_logits # this is dumb

    def get_preds(self, **kwargs):
        H,N,C = self.oracle.dataset.pred_logits.shape
        labels = self.oracle.labels
        one_hot = torch.nn.functional.one_hot(labels, num_classes=C).float()
        print("one hot", one_hot.shape, one_hot)
        return one_hot