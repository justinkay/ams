import argparse
import datasets
import labelers
from surrogates import Ensemble
from models import IID, AMS
import random
import numpy as np
import torch
from torch.nn.functional import cross_entropy
from functools import partial


DATASETS = {
    'domainnet126': datasets.DomainNet126,
}

LOSSES = {
    'domainnet126': cross_entropy,
}

LABELERS = {
    'oracle': labelers.Oracle
}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def compute_LUREs(all_preds, d_l_idxs, d_l_ys, qm, loss_fn):
    """
        all_preds: shape (num_models, num_data_points, num_classes)
        d_l_idxs: labeled indices, list of length M
        d_l_ys: corresponding labels for d_l_idxs, also length M
        qm: sampling probability for all points in d_l_idxs, also length M
    """
    H, N, C = all_preds.shape
    
    # get preds for labeled points only
    all_preds = all_preds[:, d_l_idxs, :]
    _, M, _ = all_preds.shape
    losses = []

    for h in range(H):
        loss = loss_fn(torch.tensor(all_preds[h]), torch.tensor(d_l_ys), reduction='none')
        for m in range(M):
            v_m = 1 + (N - M) / (N - m) * ( 1 / ( (N - m + 1) * qm[m] ) - 1)
            loss[m] *= v_m
        losses.append(loss.sum() / M)

    return losses

def ams_acquisition_with_p(d_u, all_preds, p_h=None):
    pass


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
    print(args)
    seed_all(args.seed)

    # load prediction results of all hypotheses
    dataset = DATASETS[args.dataset](args.task)
    dataset.load_runs(subsample_pct=args.subsample_pct)

    # for now, dataset-specific loss functions
    loss_fn = LOSSES[args.dataset]

    # model selection algorithm
    if args.algorithm == 'ours':
        surrogate = Ensemble(dataset.preds)
        model = AMS(surrogate, loss_fn)
    elif args.algorithm == 'iid':
        model = IID()
    else:
        raise NotImplementedError("Algorithm" + args.algorithm + "not supported.")
    
    # label function - oracle or user
    labeler = LABELERS[args.labeler](dataset)
    
    # labeled data point indices and labels - at first, no points are labeled
    d_l_idxs = [] 
    d_l_ys = []

    # unlabeled data point indices - at first, all points are unlabeled
    d_u_idxs = list(range(dataset.preds.shape[1]))

    # store acquisition probabilities for labeled points to compute LURE estimates
    qms = [] 

    # active model selection loop
    for m in range(args.iters):
        # sample data point
        d_m_idx, qm = model.do_step(d_u_idxs)
        d_l_idxs.append(d_m_idx)
        d_u_idxs.remove(d_m_idx)
        qms.append(qm)

        # label data point
        d_m_y = labeler(d_m_idx)
        d_l_ys.append(d_m_y)

        # update loss estimates for all models
        all_LURE_estimates = compute_LUREs(dataset.preds, d_l_idxs, d_l_ys, qms, loss_fn)

        # do model selection
        best_model_idx_pred = np.argmin(np.array(all_LURE_estimates))
        best_loss_pred = all_LURE_estimates[best_model_idx_pred]
        
        print("iter", m)
        print("Best model idx pred", best_model_idx_pred)
        print("Best loss pred", best_loss_pred)
        print()

        # log results
        # predicted_idxs.append(best_model_idx_pred)
        # predicted_best_losses.append(best_loss_pred)
        # true_best_losses.append(true_losses[best_model_idx_pred])
        # true_best_accs.append(true_accs[best_model_idx_pred])

if __name__ == "__main__":
    main()