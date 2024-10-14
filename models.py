import numpy as np
import torch
from scipy.stats import norm

class AMS:
    def __init__(self, surrogate, loss_fn, prune_thresh=0.05):
        self.surrogate = surrogate
        self.loss_fn = loss_fn
        self.prune_thresh = prune_thresh # every iteration, prune models whose P(h=h*) < self.prune_thresh

    def _losses(self):
        all_preds_tensor = torch.tensor(self.surrogate.all_preds)
        H, N, C = all_preds_tensor.shape
        surrogate_preds_tensor = self.surrogate.get_preds()

        loss = self.loss_fn(
            all_preds_tensor.view(H * N, C), 
            surrogate_preds_tensor.repeat(H, 1), 
            reduction='none'
        ).view(H, N)  # Shape: (H, N)
        return loss
    
    def _get_loss_mean_variance(self, loss):
        # Calculate the mean loss (mu_h) for each model
        surrogate_mu = loss.mean(dim=1).cpu().numpy()  # Shape: (H,)

        # Calculate variance and derive the shared standard deviation (sigma)
        variance = loss.var(dim=0).mean().item()  # Compute variance across data points
        surrogate_sigma = np.sqrt(variance)  # Shared standard deviation

        return surrogate_mu, surrogate_sigma
    
    def _get_p_h(self, surrogate_mu, surrogate_sigma):
        diff_matrix = surrogate_mu[:, None] - surrogate_mu[None, :]  # Shape: (H, H)
        pairwise_probs = norm.cdf(diff_matrix / (np.sqrt(2) * surrogate_sigma))  # Shape: (H, H)
        np.fill_diagonal(pairwise_probs, 1)  # Set diagonal to 1 as each model is equally likely compared to itself
        p_h = pairwise_probs.prod(axis=1)  # Shape: (H,)

        # Normalize p_h to sum to 1
        p_h /= p_h.sum()

        return p_h

    def do_step(self, d_u_idxs):
        # get losses of *each model* based on surrogate model
        loss = self._losses()

        # compute the mean loss and variance for *each data point*
        surrogate_mu, surrogate_sigma = self._get_loss_mean_variance(loss)

        # compute prob of each model being the best
        p_h = self._get_p_h(surrogate_mu, surrogate_sigma)
        p_h_tensor = torch.tensor(p_h, dtype=torch.float32)

        # get sampling probability as p(h=h*) x L(h)
        weighted_losses = (p_h_tensor[:, None] * loss).sum(dim=0)  # Shape: (N,)
        qs = weighted_losses[d_u_idxs].cpu().numpy()

        # Replace NaNs with zeros and ensure the sum of qs is non-zero before normalizing
        qs = np.nan_to_num(qs, nan=0.0)
        # make sure all points have a chance of being sampled
        qs[qs <= 0] = 1e-9

        if qs.sum() == 0:
            # If all values are zero, make a uniform distribution over d_u
            qs = np.ones_like(qs) / len(qs)
        else:
            # Normalize qs to form a probability distribution
            qs = qs / qs.sum()

        # Sample a point using the calculated probabilities
        choice = np.random.choice(d_u_idxs, p=qs)

        # TODO: prune surrogate based on p_h

        return choice, qs[d_u_idxs.index(choice)]

class IID:
    pass