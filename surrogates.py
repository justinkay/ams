class Ensemble:
    def __init__(self, all_preds):
        self.all_preds = all_preds
        self.device = all_preds.device
        H, N, C = all_preds.shape

    def get_preds(self, **kwargs):
        return self.all_preds.mean(dim=0)

class WeightedEnsemble(Ensemble):
    def __init__(self, all_preds):
        super().__init__(all_preds)

    def get_preds(self, weights):
        return (self.all_preds * weights.view(-1, 1, 1)).sum(dim=0)