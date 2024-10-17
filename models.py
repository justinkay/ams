import random
import torch
from torch.distributions import Normal

class AMS:
    def __init__(self, surrogate, loss_fn, prune_thresh=0.05, batch_size=1000):
        self.surrogate = surrogate
        self.loss_fn = loss_fn
        self.prune_thresh = prune_thresh
        self.device = surrogate.device
        self.batch_size = batch_size

    def _losses(self):
        all_preds_tensor = self.surrogate.all_preds
        H, N, C = all_preds_tensor.shape
        surrogate_preds_tensor = self.surrogate.get_preds()

        losses = torch.zeros(H, N, device=self.device)

        for i in range(0, N, self.batch_size):
            batch_end = min(i + self.batch_size, N)
            batch_preds = all_preds_tensor[:, i:batch_end, :].reshape(-1, C)
            batch_surrogate = surrogate_preds_tensor[i:batch_end].repeat(H, 1)

            batch_loss = self.loss_fn(
                batch_preds,
                batch_surrogate,
                reduction='none'
            ).view(H, -1)

            losses[:, i:batch_end] = batch_loss

        return losses
    
    def _get_loss_mean_variance(self, loss):
        surrogate_mu = loss.mean(dim=1)
        variance = loss.var(dim=0).mean()
        surrogate_sigma = torch.sqrt(variance)
        return surrogate_mu, surrogate_sigma
    
    def _get_p_h(self, surrogate_mu, surrogate_sigma):
        diff_matrix = surrogate_mu.unsqueeze(1) - surrogate_mu.unsqueeze(0)
        normal = Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        pairwise_probs = normal.cdf(diff_matrix / (torch.sqrt(torch.tensor(2.0, device=self.device)) * surrogate_sigma))
        pairwise_probs.fill_diagonal_(1)
        log_p_h = pairwise_probs.log().sum(dim=1)
        p_h = torch.exp(log_p_h - torch.logsumexp(log_p_h, dim=0))
        return p_h

    def do_step(self, d_u_idxs):
        loss = self._losses()
        surrogate_mu, surrogate_sigma = self._get_loss_mean_variance(loss)
        p_h = self._get_p_h(surrogate_mu, surrogate_sigma)

        weighted_losses = (p_h.unsqueeze(1) * loss).sum(dim=0)
        qs = weighted_losses[d_u_idxs]

        qs = torch.nan_to_num(qs, nan=0.0)
        qs[qs <= 0] = 1e-9

        if qs.sum() == 0:
            qs = torch.ones_like(qs) / len(qs)
        else:
            qs = qs / qs.sum()

        choice = torch.multinomial(qs, 1).item()
        return d_u_idxs[choice], qs[choice].item()

class IID:
    def __init__(self):
        pass

    def do_step(self, d_u_idxs):
        choice = random.randint(0, len(d_u_idxs) - 1)
        q = 1.0 / len(d_u_idxs)
        return d_u_idxs[choice], q
