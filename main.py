from comet_ml import Experiment
import argparse
import datasets
import labelers
from surrogates import Ensemble, WeightedEnsemble
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

def compute_LUREs(all_preds, d_l_idxs, d_l_ys, qm, loss_fn, device, batch_size=100):
    """
        all_preds: shape (num_models, num_data_points, num_classes)
        d_l_idxs: labeled indices, list of length M
        d_l_ys: corresponding labels for d_l_idxs, also length M
        qm: sampling probability for all points in d_l_idxs, also length M
    """
    H, N, C = all_preds.shape
    M = len(d_l_idxs)
    
    # Move all data to the specified device
    all_preds = all_preds[:, d_l_idxs, :].to(device)
    d_l_ys = torch.tensor(d_l_ys, device=device)
    qm = torch.tensor(qm, device=device)

    losses = torch.zeros(H, device=device)

    for i in range(0, M, batch_size):
        batch_end = min(i + batch_size, M)
        batch_preds = all_preds[:, i:batch_end, :]
        batch_labels = d_l_ys[i:batch_end]
        batch_qm = qm[i:batch_end]

        batch_losses = loss_fn(batch_preds.reshape(-1, C), batch_labels.repeat(H), reduction='none').view(H, -1)

        for m in range(batch_end - i):
            v_m = 1 + (N - M) / (N - (i + m)) * (1 / ((N - (i + m) + 1) * batch_qm[m]) - 1)
            batch_losses[:, m] *= v_m

        losses += batch_losses.sum(dim=1)

    return (losses / M).tolist()

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
    parser.add_argument("--ensemble", default="naive", help="{ 'naive', 'weighted', ... }")
    parser.add_argument("--force-reload", action='store_true', help="Load directly from feature files rather than large dat file.")
    parser.add_argument("--no-write", action='store_true', help="Don't write preds to an intermediate .dat or .pt file.")
    parser.add_argument("--no-comet", action='store_true', help="Disable logging with Comet ML")

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()

def main():
    args = parse_args()
    seed_all(args.seed)
    
    # modify args as needed
    if args.algorithm == 'iid':
        args.ensemble = 'none'

    experiment = Experiment(
        project_name="ams",
        workspace=os.environ["COMET_WORKSPACE"],
        api_key=os.environ["COMET_API_KEY"],
        log_env_details=False,
        disabled = args.no_comet
    )
    experiment.set_name("-".join([f'{k}={str(v)}' for k,v in vars(args).items()]))
    experiment.log_parameters(args)
    experiment.log_parameter("algorithm-detail", "-".join([args.algorithm, args.ensemble, args.loss]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load prediction results of all hypotheses
    dataset = DATASETS[args.dataset](args.task, dataset_filter=args.dataset_filter)
    dataset.load_runs(subsample_pct=args.subsample_pct, force_reload=args.force_reload, write=not args.no_write)
    experiment.log_parameter("num_runs", len(dataset.get_run_tasks()))

    # loss and accuracy functions
    accuracy_fn = ACCURACY_FNS[args.dataset].to(dataset.device)
    loss_fn = LOSS_FNS[args.loss]

    # model selection algorithm
    if args.algorithm == 'ours':
        if args.ensemble == 'naive':
            surrogate = Ensemble(dataset.preds)
        elif args.ensemble == 'weighted':
            surrogate = WeightedEnsemble(dataset.preds)
        elif args.ensemble != 'none':
            raise NotImplementedError("Ensemble" + args.ensemble + "not supported.")
        model = AMS(surrogate, loss_fn)
    elif args.algorithm == 'iid':
        model = IID()
    else:
        raise NotImplementedError("Algorithm" + args.algorithm + "not supported.")
    
    # label function - oracle or user
    oracle = labelers.Oracle(dataset, loss_fn=loss_fn, accuracy_fn=accuracy_fn)
    if args.labeler == 'oracle':
        labeler = oracle
    else:
        labeler = LABELERS[args.labeler]

    # log our baselines/best case results
    best_loss = min(oracle.true_losses)
    best_acc =  max(oracle.true_accs)
    experiment.log_metric("Best loss", best_loss)
    experiment.log_metric("Best accuracy", best_acc)
    
    print("min and max loss",min(oracle.true_losses),max(oracle.true_losses))
    print("min and max accuracy", min(oracle.true_accs), max(oracle.true_accs))

    # labeled data point indices and labels - at first, no points are labeled
    d_l_idxs = [] 
    d_l_ys = []

    # unlabeled data point indices - at first, all points are unlabeled
    d_u_idxs = list(range(dataset.preds.shape[1]))

    # store acquisition probabilities for labeled points to compute LURE estimates
    qms = []

    # metrics we will track
    cumulative_regret_loss = 0
    cumulative_regret_acc = 0

    # active model selection loop
    for m in tqdm(range(args.iters)):
        # sample data point
        print("doing step")
        d_m_idx, qm = model.do_step(d_u_idxs)

        d_l_idxs.append(d_m_idx)
        d_u_idxs.remove(d_m_idx)
        qms.append(qm)

        # label data point
        d_m_y = labeler(d_m_idx)
        d_l_ys.append(d_m_y)

        # update loss estimates for all models
        all_LURE_estimates = compute_LUREs(dataset.preds, d_l_idxs, d_l_ys, qms, loss_fn, device)

        # do model selection
        best_model_idx_pred = np.argmin(np.array(all_LURE_estimates))
        best_loss_pred = all_LURE_estimates[best_model_idx_pred]
        
        # log metrics
        experiment.log_metric("Pred. best model idx", best_model_idx_pred, step=m)
        experiment.log_metric("Pred. best model, pred. loss", best_loss_pred, step=m)
        experiment.log_metric("Pred. best model, true loss", oracle.true_losses[best_model_idx_pred], step=m)
        experiment.log_metric("Pred. best model, true accuracy", oracle.true_accs[best_model_idx_pred], step=m)

        cumulative_regret_loss += oracle.true_losses[best_model_idx_pred] - best_loss
        cumulative_regret_acc += best_acc - oracle.true_accs[best_model_idx_pred]
        experiment.log_metric("Cumulative regret (loss)", cumulative_regret_loss, step=m)
        experiment.log_metric("Cumulative regret (acc)", cumulative_regret_acc, step=m)

        # get ensemble accuracy at this time step
        if args.ensemble == 'weighted':
            ensemble_preds = surrogate.get_preds(weights=model.last_p_h)
        elif args.ensemble == 'naive':
            ensemble_preds = surrogate.get_preds()
        if args.ensemble != 'none':
            N, C = ensemble_preds.shape
            ensemble_loss = loss_fn(ensemble_preds.reshape(-1, C), oracle.labels).mean() # TODO is this the right way to mean()
            ensemble_acc = accuracy_fn(ensemble_preds.reshape(-1, C), oracle.labels)
            experiment.log_metric("Ensemble loss", ensemble_loss, step=m)
            experiment.log_metric("Ensemble accuracy", ensemble_acc, step=m)

if __name__ == "__main__":
    main()