from comet_ml import Experiment
import argparse
import datasets
import labelers
from surrogates import Ensemble
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

LOSS_FNS = {
    'domainnet126': cross_entropy,
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
    parser.add_argument("--task", help="{ 'sketch_painting', ... }", default=None)
    parser.add_argument("--labeler", help="{ 'oracle', 'user' }", default="oracle")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--algorithm", help="{ 'iid', 'ours', 'ours-sfo', ... }", default="ours")
    parser.add_argument("--subsample_pct", type=int, help="Percentage of runs to analyze",  default=100)
    parser.add_argument("--seed", default=0)

    # other potential options:
    # loss fn (cross-entropy vs. accuracy, e.g.)
    # --no-prune; other ablation studies

    return parser.parse_args()

def main():
    args = parse_args()
    seed_all(args.seed)
    
    experiment = Experiment(
        project_name="ams",
        workspace=os.environ["COMET_WORKSPACE"],
        api_key=os.environ["COMET_API_KEY"],
        log_env_details=False
    )
    experiment.set_name("-".join([f'{k}={str(v)}' for k,v in vars(args).items()]))
    experiment.log_parameters(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load prediction results of all hypotheses
    dataset = DATASETS[args.dataset](args.task)
    dataset.load_runs(subsample_pct=args.subsample_pct)

    # for now, dataset-specific loss functions
    loss_fn = LOSS_FNS[args.dataset]
    accuracy_fn = ACCURACY_FNS[args.dataset].to(dataset.device)

    # model selection algorithm
    if args.algorithm == 'ours':
        surrogate = Ensemble(dataset.preds)
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

        # log other stuff
        # P(h=h*)
        # etc...

if __name__ == "__main__":
    main()