import random
import torch
from torch.distributions import Normal


class AMS:
    def __init__(self, surrogate, loss_fn, batch_size=1000, use_p_h=True):
        self.surrogate = surrogate
        self.loss_fn = loss_fn
        self.device = surrogate.device
        self.batch_size = batch_size
        self.use_p_h = use_p_h # utilize estimates of p(h=h*); if False, keep this as a uniform distribution

        # initial belief over hypotheses
        self.last_p_h = torch.ones((surrogate.pred_logits.shape[0],), device=self.device) / surrogate.pred_logits.shape[0]

    def _losses(self):
        pred_logits = self.surrogate.pred_logits
        H, N, C = pred_logits.shape
        print('h n c', H, N, C)

        # this is assumed to be softmaxed already
        surrogate_preds_tensor = self.surrogate.get_preds(weights=self.last_p_h)
        print("surrogate preds tensor is softmaxed correctly?", surrogate_preds_tensor.shape, surrogate_preds_tensor.sum(dim=-1))

        losses = torch.zeros(H, N, device=self.device)
        for i in range(0, N, self.batch_size):
            batch_end = min(i + self.batch_size, N)
            batch_preds = pred_logits[:, i:batch_end, :].reshape(-1, C)
            print("batch preds", batch_preds.shape, batch_preds)
            batch_surrogate = surrogate_preds_tensor[i:batch_end].repeat(H, 1) # ground truth on these points is same for every model
            print("batch surrogate", batch_surrogate.shape, batch_surrogate)
            batch_loss = self.loss_fn( 
                batch_preds,
                batch_surrogate,
                reduction='none'
            ).view(H, -1)

            losses[:, i:batch_end] = batch_loss

        print("hypotheses losses", losses.shape, losses)

        return losses
    
    def _get_loss_mean_variance(self, loss):
        """Get the mean loss and variance (over the whole dataset) for each hypotheses.
        Args:
            loss: shape [H,N]
        """
        mu_h = loss.mean(dim=1)
        print("surrogate_mu.shape", mu_h.shape)

        # mu_h is shape [H]
        variance = loss.var(dim=0).mean()
        sigma = torch.sqrt(variance)

        return mu_h, sigma
    
    def _compute_p_h(self, mu_h, sigma):
        """Compute p(h=h*), i.e. the probability that the loss of model h is the lowest of all models.
        We assume each model's loss is normally distriuted ~ N(mu_h[h], sigma).
        Args:
            mu_h: shape (H,), one (estimated) loss for each model, which we assume
            sigma: one shared std dev for all models
        """
        diff_matrix = mu_h.unsqueeze(1) - mu_h.unsqueeze(0)
        normal = Normal(torch.tensor([0.0], device=self.device), torch.tensor([1.0], device=self.device))
        pairwise_probs = normal.cdf(diff_matrix / (torch.sqrt(torch.tensor(2.0, device=self.device)) * sigma))
        pairwise_probs.fill_diagonal_(1)
        log_p_h = pairwise_probs.log().sum(dim=1)
        p_h = torch.exp(log_p_h - torch.logsumexp(log_p_h, dim=0))
        self.last_p_h = p_h
        return self.last_p_h

    def do_step(self, d_u_idxs):
        print("getting losses wrt surrogate")
        loss = self._losses() # shape (H, N)

        if self.use_p_h:
            # mu_h: shape (H,) - mean loss of each hypothesis over whole dataset, w.r.t. surrogate pseudo ground truth
            # sigma: shape (,) - one variance value over all losses over all hypotheses
            mu_h, sigma = self._get_loss_mean_variance(loss)
            print("mu_h", mu_h.shape, mu_h)
            print("sigma", sigma.shape, sigma)
            p_h = self._compute_p_h(mu_h, sigma)
            print("p_h", p_h.shape, p_h)
            print("p_h min, max, mean", p_h.min(), p_h.max(), p_h.mean())

        print("losses", loss.shape, loss)
        weighted_losses = (self.last_p_h.unsqueeze(1) * loss).sum(dim=0)
        print("weighted losses", weighted_losses.shape, weighted_losses)

        print("subtracting min loss")
        weighted_losses = weighted_losses - weighted_losses.min()
        print("weighted losses min subtracted", weighted_losses.shape, weighted_losses)

        qs = weighted_losses[d_u_idxs]

        print("qs0", qs)
        qs = torch.nan_to_num(qs, nan=0.0)
        print("qs1", qs)

        # TODO should probably assert nothing below 0
        qs[qs <= 0] = 1e-9
        print("qs2", qs)

        if qs.sum() == 0:
            print("qs sum is 0")
            qs = torch.ones_like(qs) / len(qs)
        else:
            print("qs sum is not 0")
            print("qs sum", qs.sum())
            print("qs before norm: shape, min, max, mean, std", qs.shape, qs.min(), qs.max(), qs.mean(), qs.std())
            qs = qs / qs.sum()

        print("qs final: shape, min, max, mean, std", qs.shape, qs.min(), qs.max(), qs.mean(), qs.std())

        import matplotlib.pyplot as plt
        plt.bar(list(range(len(qs))), qs.cpu().numpy())
        plt.savefig("qs.png")
        
        choice = torch.multinomial(qs, 1).item()
        return d_u_idxs[choice], qs[choice].item()

class IID:
    def __init__(self):
        pass

    def do_step(self, d_u_idxs):
        choice = random.randint(0, len(d_u_idxs) - 1)
        q = 1.0 / len(d_u_idxs)
        return d_u_idxs[choice], q
