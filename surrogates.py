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
        # print("ensemble weights", weights.shape, weights, weights.sum())
        # print("weights view", weights.view(-1, 1, 1).shape, weights.view(-1, 1, 1))
        # print("softmaxed logits", torch.softmax(self.pred_logits, dim=-1).shape, torch.softmax(self.pred_logits, dim=-1))
        # print("after mult", (torch.softmax(self.pred_logits, dim=-1) * weights.view(-1, 1, 1)).shape, (torch.softmax(self.pred_logits, dim=-1) * weights.view(-1, 1, 1)))
        # print("After sum", (torch.softmax(self.pred_logits, dim=-1) * weights.view(-1, 1, 1)).sum(dim=0).shape, (torch.softmax(self.pred_logits, dim=-1) * weights.view(-1, 1, 1)).sum(dim=0))
        # return (torch.softmax(self.pred_logits, dim=-1) * weights.view(-1, 1, 1)).sum(dim=0)

        # alternative that seems to work better- this could more accurately be called a 'FilteredEnsemble':
        nonzeros = torch.nonzero(weights).squeeze()
        print("nonzero weights", nonzeros.shape)
        selected_pred_logits = self.pred_logits[nonzeros, :, :]
        return torch.softmax(selected_pred_logits, dim=-1).mean(dim=0)

    
class OracleSurrogate:
    def __init__(self, oracle):
        self.oracle = oracle
        self.device = oracle.device
        self.pred_logits = self.oracle.dataset.pred_logits # this is dumb

    def get_preds(self, **kwargs):
        H,N,C = self.oracle.dataset.pred_logits.shape
        labels = self.oracle.labels
        one_hot = torch.nn.functional.one_hot(labels, num_classes=C).float()
        return one_hot