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

        # initial belief over hypotheses
        self.last_p_h = torch.ones((surrogate.all_preds.shape[0],), device=self.device) / surrogate.all_preds.shape[0]

    def _losses(self):
        all_preds_tensor = self.surrogate.all_preds
        H, N, C = all_preds_tensor.shape
        surrogate_preds_tensor = self.surrogate.get_preds(weights=self.last_p_h)
        print("surrogate preds", surrogate_preds_tensor.shape, surrogate_preds_tensor)

        losses = torch.zeros(H, N, device=self.device)

        # get losses between our preds and surrogate preds
        # batch up the data to reduce GPU memory load -- compute all model losses on self.batch_size data points at a time
        for i in range(0, N, self.batch_size):
            batch_end = min(i + self.batch_size, N)
            batch_preds = all_preds_tensor[:, i:batch_end, :].reshape(-1, C)
            batch_surrogate = torch.softmax(surrogate_preds_tensor[i:batch_end], dim=-1).repeat(H, 1) # ground truth on these points is same for every model
            batch_loss = self.loss_fn( 
                batch_preds,
                batch_surrogate, # TODO the softmax here may not generalize to other tasks/losses
                reduction='none'
            ).view(H, -1)

            if i == 0:
                print("iter", i)
                print("batch_preds", batch_preds.shape, batch_preds)
                print("batch_surrogate", batch_surrogate.shape, batch_surrogate)
                print("batch_loss", batch_loss.shape, batch_loss)

            losses[:, i:batch_end] = batch_loss

        print("surrogate losses", losses.shape, losses)

        return losses
    
    def _get_loss_mean_variance(self, loss):
        surrogate_mu = loss.mean(dim=1)
        variance = loss.var(dim=0).mean()
        surrogate_sigma = torch.sqrt(variance)
        return surrogate_mu, surrogate_sigma
    
    def _compute_p_h(self, surrogate_mu, surrogate_sigma):
        diff_matrix = surrogate_mu.unsqueeze(1) - surrogate_mu.unsqueeze(0)
        normal = Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        pairwise_probs = normal.cdf(diff_matrix / (torch.sqrt(torch.tensor(2.0, device=self.device)) * surrogate_sigma))
        pairwise_probs.fill_diagonal_(1)
        log_p_h = pairwise_probs.log().sum(dim=1)
        p_h = torch.exp(log_p_h - torch.logsumexp(log_p_h, dim=0))
        self.last_p_h = p_h
        return self.last_p_h

    def do_step(self, d_u_idxs):
        print("getting surrogate losses")
        loss = self._losses()

        surrogate_mu, surrogate_sigma = self._get_loss_mean_variance(loss)
        print("surrogate_mu", surrogate_mu.shape, surrogate_mu)
        print("surrogate_sigma", surrogate_sigma.shape, surrogate_sigma)

        p_h = self._compute_p_h(surrogate_mu, surrogate_sigma)

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
