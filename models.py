import random
import torch
from torch.distributions import Normal


class AMS:
    def __init__(self, surrogate, loss_fn, lure_estimator, batch_size=1000, use_p_h=True):
        self.surrogate = surrogate
        self.loss_fn = loss_fn
        self.device = surrogate.device
        self.batch_size = batch_size
        self.lure_estimator = lure_estimator

        # Utilize estimates of p(h=h*); if False, keep this as a uniform distribution
        self.use_p_h = use_p_h 

        # Initial belief over hypotheses
        self.last_p_h = torch.ones((surrogate.pred_logits.shape[0],), device=self.device) / surrogate.pred_logits.shape[0]

    def _losses(self):
        """
        Return:
            losses: shape (H, N) -- estimated losses for each hypothesis for each data point
        """
        pred_logits = self.surrogate.pred_logits
        H, N, C = pred_logits.shape

        # this is assumed to be softmaxed already
        surrogate_preds_tensor = self.surrogate.get_preds(weights=self.last_p_h)

        losses = torch.zeros(H, N, device=self.device)
        for i in range(0, N, self.batch_size):
            batch_end = min(i + self.batch_size, N)
            batch_preds = pred_logits[:, i:batch_end, :].reshape(-1, C)
            batch_surrogate = surrogate_preds_tensor[i:batch_end].repeat(H, 1) # ground truth on these points is same for every model
            batch_loss = self.loss_fn( 
                batch_preds,
                batch_surrogate,
                reduction='none'
            ).view(H, -1)

            losses[:, i:batch_end] = batch_loss

        return losses

    # TODO: check chatgpt
    def _compute_p_h(self, mu_h, sigma_h):
        """
        Compute p(h=h*), i.e., the probability that the loss of model h is the lowest of all models.
        We assume each model's loss is normally distributed ~ N(mu_h[h], sigma_h[h]).
        
        Args:
            mu_h: shape (H,), one (estimated) mean loss for each model.
            sigma_h: shape (H,), standard deviation for each model's loss.
        """
        H = mu_h.shape[0]  # Number of models
        
        # Create a matrix where each element (i,j) contains the difference mu_h[i] - mu_h[j]
        diff_matrix = mu_h.unsqueeze(1) - mu_h.unsqueeze(0)  # Shape (H, H)
        
        # Compute the pairwise standard deviations for each pair of models
        sigma_matrix = torch.sqrt(sigma_h.unsqueeze(1)**2 + sigma_h.unsqueeze(0)**2)  # Shape (H, H)
        
        # Normal distribution with mean 0 and variance 1
        normal = Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        
        # Compute the pairwise probabilities using the individual sigma for each model pair
        pairwise_probs = normal.cdf(diff_matrix / (torch.sqrt(torch.tensor(2.0, device=self.device)) * sigma_matrix))
        
        # Set the diagonal to 1 (since we compare a model with itself)
        pairwise_probs.fill_diagonal_(1)
        
        # Compute the log of the probabilities for numerical stability
        log_p_h = pairwise_probs.log().sum(dim=1)  # Log probability that each model has the lowest loss
        
        # Convert back to probabilities and normalize
        p_h = torch.exp(log_p_h - torch.logsumexp(log_p_h, dim=0))  # Normalize to ensure the probabilities sum to 1
        
        # Store the last p_h and return
        self.last_p_h = p_h

    def do_step(self, d_u_idxs):
        loss = self._losses() # shape (H, N); uses self.last_p_h as necessary

        if self.use_p_h and self.lure_estimator.M >= 2: # need at least 2 points to get variance
            # mus: shape (H,) - mean loss of each hypothesis over whole dataset (LURE estimate)
            # sigmas: shape (H,) - variance of LURE estimates
            # mu_h, sigma = self._get_loss_mean_variance(loss)
            mus, sigmas = self.lure_estimator.get_LUREs_and_vars()
            
            # use these to compute p(h=h*), probability that each model is the best
            # (stored in self.last_p_h)
            self._compute_p_h(mus, sigmas)

        # weight losses w.r.t. ranking of hypotheses
        weighted_losses = (self.last_p_h.unsqueeze(1) * loss).sum(dim=0)
        weighted_losses = weighted_losses - weighted_losses.min() # new approach
        qs = weighted_losses[d_u_idxs]

        # fix up qs to make sure it is a valid distribution
        # TODO should probably assert nothing below 0
        qs = torch.nan_to_num(qs, nan=0.0)
        qs[qs <= 0] = 1e-9
        if qs.sum() == 0:
            qs = torch.ones_like(qs) / len(qs)
        else:
            qs = qs / qs.sum()

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.bar(list(range(len(qs))), qs.cpu().numpy())
        # plt.savefig(f"qs-{self.lure_estimator.M}.png")
        
        choice = torch.multinomial(qs, 1).item()
        return d_u_idxs[choice], qs[choice].item()

class IID:
    def __init__(self):
        pass

    def do_step(self, d_u_idxs):
        choice = random.randint(0, len(d_u_idxs) - 1)
        q = 1.0 / len(d_u_idxs)
        return d_u_idxs[choice], q
