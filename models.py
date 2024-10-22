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

    def _compute_p_h(self, mu_h, sigma_h_squared):
        """
        Compute p(h=h*), i.e., the probability that the loss of model h is the lowest of all models.
        We assume each model's loss is normally distributed ~ N(mu_h[h], sigma_h[h]).
        
        Args:
            mu_h: shape (H,), one (estimated) mean loss for each model.
            sigma_h: shape (H,), standard deviation for each model's loss.
        """
        # # gumbel max trick
        # num_samples = 20000
        # H = mu_h.shape[0]

        # print("mu_h", mu_h)
        # print("mu_h min", mu_h.min())
        # print("sigma squared", sigma_h_squared)
        # print("sigma squared min", sigma_h_squared.min())

        # # variance -> std
        # sigma_h = torch.sqrt(sigma_h_squared)
    
        # # Sample from Gumbel distribution
        # gumbel_noise = -torch.log(-torch.log(torch.rand(H, num_samples, device=mu_h.device)))

        # # Sample from the Gaussian loss distribution and add Gumbel noise
        # losses_with_noise = mu_h.unsqueeze(1) + sigma_h.unsqueeze(1) * torch.randn(H, num_samples, device=mu_h.device) + gumbel_noise
        # print("losses with noise", losses_with_noise.shape, losses_with_noise)

        # # Determine the model with the lowest (noisy) loss for each sample
        # best_model_losses, best_model_idxs = torch.min(losses_with_noise, dim=0)
        # print("best model_idxs", best_model_idxs.shape, best_model_idxs)
        # print("best losses", best_model_losses.shape, best_model_losses)
        
        # # Count how often each model is the best
        # best_model_counts = torch.bincount(best_model_idxs, minlength=H)
        # print("best_model_counts",best_model_counts)
        
        # # Convert counts to probabilities
        # p_h = best_model_counts.float() / num_samples
        
        # self.last_p_h = p_h
        # return p_h

        # monte carlo method
        num_samples = 10000
        H = mu_h.shape[0]

        # Compute standard deviations from the variances
        sigma_h = torch.sqrt(sigma_h_squared)

        # Identify the device
        device = mu_h.device  # Assuming mu_h and sigma_h are on the same device

        # Define batch size based on available memory
        batch_size = 1000  # Adjust this number based on your memory constraints
        num_batches = (num_samples + batch_size - 1) // batch_size

        # Initialize counts on the correct device
        best_model_counts = torch.zeros(H, dtype=torch.int64, device=device)

        remaining_samples = num_samples  # Keep track of remaining samples

        for _ in range(num_batches):
            current_batch_size = min(batch_size, remaining_samples)
            remaining_samples -= current_batch_size

            # Generate samples on the correct device
            samples = torch.randn(H, current_batch_size, device=device) * sigma_h[:, None] + mu_h[:, None]

            # Determine which model has the lowest loss for each sample
            best_model_idxs = torch.argmin(samples, dim=0)

            # Count how often each model is the best
            batch_counts = torch.bincount(best_model_idxs, minlength=H)

            best_model_counts += batch_counts

        total_samples = num_samples - remaining_samples
        # Convert counts to probabilities
        p_h = best_model_counts.float() / total_samples

        self.last_p_h = p_h
        return p_h

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
        weighted_losses = weighted_losses - weighted_losses.min() # new approach - scale q to min loss at each point
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
