from comet_ml import Experiment
import argparse
import datasets
import labelers
from surrogates import Ensemble, WeightedEnsemble, OracleSurrogate
from models import IID, AMS
import random
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from torchmetrics import Accuracy
from tqdm import tqdm
import os


DATASETS = {
    'domainnet126': datasets.DomainNet126,
}

def accuracy_loss(preds, labels, **kwargs):
    """Get 1 - accuracy (a loss), nonreduced. Handles whether we are working with scores or integer labels."""
    if len(labels.shape) > 1:
        argmaxed_preds = torch.argmax(preds, dim=-1)
        argmaxed_labels = torch.argmax(labels, dim=-1)
        accs = (argmaxed_preds == argmaxed_labels).float()
    else:
        argmaxed = torch.argmax(preds, dim=-1)
        accs = (argmaxed == labels).float()

    # make it a loss
    return 1 - accs

LOSS_FNS = {
    'ce': cross_entropy,
    'acc': accuracy_loss
}

LABELERS = {
    'oracle': labelers.Oracle
}

ACCURACY_FNS = {
    'domainnet126': Accuracy(task="multiclass", num_classes=126, average="macro"), # Musgrave et al use macro average
}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class LUREEstimator:
    def __init__(self, H, N, C, loss_fn, device):
        """
        H: Number of models in the ensemble
        N: Total number of data points
        C: Number of classes
        loss_fn: Loss function to compute losses between predictions and true labels
        device: The device (CPU/GPU) to perform computations
        """
        self.H = H
        self.N = N
        self.C = C
        self.loss_fn = loss_fn
        self.device = device

        # Actively sampled points
        self.M = 0  # Number of sampled points
        self.losses = []  # True losses for each model - shape (H, M)
        self.qs = []  # Sampling probabilities for each point - shape (M,)

    def get_vs(self):
        """
        Compute the LURE weights (v_m) for each sampled point based on the current state.
        This is called after adding a new observation to update the weights.
        """
        vs = []
        for m in range(self.M):
            v = 1 + ( (self.N - self.M) / (self.N - m) ) * (1 / ((self.N - m + 1) * self.qs[m]) - 1) 
            vs.append(v)
        return vs

    def add_observation(self, pred_logits, label, qm):
        """
        Adds a new observation with the predicted logits, true label, and sampling probability.
        
        Args:
            pred_logits: shape (H, C) - predicted logits for each model
            label: int - true label for the sampled point
            qm: float - sampling probability of this point
        """
        loss = self.loss_fn(pred_logits, torch.tensor([label], device=self.device).repeat(self.H), reduction='none')
        self.losses.append(loss)
        self.qs.append(qm)
        self.M += 1

    def get_LUREs_and_vars(self):
        """
        Compute the LURE estimates and their variances based on the current set of sampled points.
        
        Returns:
            - lure_estimates: shape (H,) - LURE estimates for each model
            - variance_lure: shape (H,) - variance of LURE estimates for each model
        """
        # Convert lists to tensors for vectorized operations
        losses = torch.stack(self.losses, dim=1).view(self.H, -1)  # Shape (H, M)
        vs = torch.tensor(self.get_vs(), device=self.device)  # Shape (M,)
        weighted_losses = vs * losses  # Shape (H, M)
        squared_losses = (vs**2) * (losses**2)  # Shape (H, M)
        lure_estimates = weighted_losses.sum(dim=1) / self.M  # Shape (H,)
        variance_lure = squared_losses.sum(dim=1) / self.M - lure_estimates**2  # Shape (H,)
        return lure_estimates, variance_lure


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="{ 'domainnet126', ... } ", required=True)
    parser.add_argument("--dataset-filter", default=None, help="Comma separated, e.g. UDA algorithms to include.")
    parser.add_argument("--task", help="{ 'sketch_painting', ... }", default=None)
    parser.add_argument("--labeler", help="{ 'oracle', 'user' }", default="oracle")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--algorithm", help="{ 'iid', 'ours', 'ours-sfo', ... }", default="ours")
    parser.add_argument("--subsample-pct", type=int, help="Percentage of runs to analyze",  default=100)
    parser.add_argument("--loss", default="ce", help="{ 'ce', 'acc', ... }")
    parser.add_argument("--ensemble", default="naive", help="{ 'naive', 'weighted', 'oracle', ... }")
    parser.add_argument("--force-reload", action='store_true', help="Load directly from feature files rather than large dat file.")
    parser.add_argument("--no-write", action='store_true', help="Don't write preds to an intermediate .dat or .pt file.")
    parser.add_argument("--no-comet", action='store_true', help="Disable logging with Comet ML")
    parser.add_argument("--no-p", action='store_true', help='Disable use of p(h=h*)')

    # how many seeds to use - one experiment per seed
    parser.add_argument("--seeds", type=int, default=3)

    return parser.parse_args()

def main():
    args = parse_args()

    # Modify args as needed
    if args.algorithm == 'iid':
        args.ensemble = 'none'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load prediction results of all hypotheses
    dataset = DATASETS[args.dataset](args.task, dataset_filter=args.dataset_filter)
    dataset.load_runs(subsample_pct=args.subsample_pct, force_reload=args.force_reload, write=not args.no_write)
    H, N, C = dataset.pred_logits.shape

    # Loss and accuracy functions
    accuracy_fn = ACCURACY_FNS[args.dataset].to(dataset.device)
    loss_fn = LOSS_FNS[args.loss]

    # Label function - oracle or (future work) a user
    oracle = labelers.Oracle(dataset, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    if args.labeler == 'oracle':
        labeler = oracle
    else:
        labeler = LABELERS[args.labeler]

    for seed in range(args.seeds):
        print("Running active model selection with seed", seed)
        seed_all(seed)

        # Setup Comet.ML experiment
        experiment = Experiment(
            project_name="ams",
            workspace=os.environ["COMET_WORKSPACE"],
            api_key=os.environ["COMET_API_KEY"],
            log_env_details=False,
            disabled = args.no_comet
        )
        experiment.set_name("-".join([f'{k}={str(v)}' for k,v in vars(args).items()]))
        experiment.log_parameters(args)
        algorithm_detail = "-".join([args.algorithm, args.ensemble, args.loss])
        if args.algorithm != 'iid':
            algorithm_detail += f"p={not args.no_p}"
        experiment.log_parameters({
            "algorithm-detail": algorithm_detail, 
            "num_runs": len(dataset.get_run_tasks()),
            "seed": seed,
        })

        # Log our baselines/best case results
        best_loss = min(oracle.true_losses)
        best_acc =  max(oracle.true_accs)
        experiment.log_metric("Best loss", best_loss)
        experiment.log_metric("Best accuracy", best_acc)
        
        # our LURE estimates and variances
        lure_estimator = LUREEstimator(H, N, C, loss_fn, device)

        # model selection algorithm
        if args.algorithm == 'ours':
            if args.ensemble == 'naive':
                surrogate = Ensemble(dataset.pred_logits)
            elif args.ensemble == 'weighted':
                surrogate = WeightedEnsemble(dataset.pred_logits)
            elif args.ensemble == 'oracle':
                surrogate = OracleSurrogate(oracle)
            elif args.ensemble != 'none':
                raise NotImplementedError("Ensemble" + args.ensemble + "not supported.")
            model = AMS(surrogate, loss_fn, lure_estimator, use_p_h=not args.no_p)
        elif args.algorithm == 'iid':
            model = IID()
        else:
            raise NotImplementedError("Algorithm" + args.algorithm + "not supported.")

        # labeled data point indices and labels - at first, no points are labeled
        d_l_idxs = [] 
        d_l_ys = []

        # unlabeled data point indices - at first, all points are unlabeled
        d_u_idxs = list(range(dataset.pred_logits.shape[1]))

        # store acquisition probabilities for labeled points to compute LURE estimates
        qms = []

        # metrics we will track
        cumulative_regret_loss = 0
        cumulative_regret_acc = 0

        # active model selection loop
        for m in tqdm(range(args.iters)):
            # sample data point
            d_m_idx, qm = model.do_step(d_u_idxs)

            d_l_idxs.append(d_m_idx)
            d_u_idxs.remove(d_m_idx)

            qms.append(qm)

            # label data point
            d_m_y = labeler(d_m_idx)
            d_l_ys.append(d_m_y)

            # update loss estimates for all models
            # LURE_means = compute_LUREs(dataset.pred_logits, d_l_idxs, d_l_ys, qms, loss_fn, device)
            # LURE_means, LURE_vars = compute_LUREs_and_variance(dataset.pred_logits, d_l_idxs, d_l_ys, qms, loss_fn, device)
            lure_estimator.add_observation(dataset.pred_logits[:, d_m_idx, :].squeeze(1), d_m_y, qm)

            # compute LURE estimates
            LURE_means, LURE_vars = lure_estimator.get_LUREs_and_vars()

            # do model selection
            best_model_idx_pred = np.argmin(np.array(LURE_means.cpu()))
            best_loss_pred = LURE_means[best_model_idx_pred]
            
            # log metrics
            experiment.log_metric("Pred. best model idx", best_model_idx_pred, step=m)
            experiment.log_metric("Pred. best model, pred. loss", best_loss_pred, step=m)
            experiment.log_metric("Pred. best model, true loss", oracle.true_losses[best_model_idx_pred], step=m)
            experiment.log_metric("Pred. best model, true accuracy", oracle.true_accs[best_model_idx_pred], step=m)

            cumulative_regret_loss += oracle.true_losses[best_model_idx_pred] - best_loss
            cumulative_regret_acc += best_acc - oracle.true_accs[best_model_idx_pred]
            experiment.log_metric("Cumulative regret (loss)", cumulative_regret_loss, step=m)
            experiment.log_metric("Cumulative regret (acc)", cumulative_regret_acc, step=m)

            # get ensemble accuracy at this time step (just for logging purposes)
            if args.ensemble != 'none': # none -> iid
                surrogate_preds = surrogate.get_preds(weights=model.last_p_h)
                N, C = surrogate_preds.shape
                surrogate_loss = loss_fn(surrogate_preds.reshape(-1, C), oracle.labels).mean() # TODO is this the right way to mean()
                surrogate_acc = accuracy_fn(surrogate_preds.reshape(-1, C), oracle.labels)
                experiment.log_metric("Ensemble loss", surrogate_loss, step=m)
                experiment.log_metric("Ensemble accuracy", surrogate_acc, step=m)

if __name__ == "__main__":
    main()