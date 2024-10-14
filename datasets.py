import os
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import random


class DomainNet126:
    """DomainNet126. 
    Assumes experiments were run with powerful-benchmarker and follow the directory structure defined there.
    """

    def __init__(self, task, base_dir="../powerful-benchmarker/experiments/", use_target_val=False):
        """
        Args:
            task: should be {source}_{target}, e.g. "sketch_painting"
            use_target_val: Whether to use the target validation set. Use the target training set if False. False by default for
                            powerful-benchmarker experiments (i.e. those written to validator_ScoreHistory.csv)
        """
        # where the experiments are saved
        self.location = os.path.join(base_dir, f"domainnet126_{task}_fl6_Adam_lr1")
        self.use_target_val = use_target_val
        self.task = task

    @staticmethod
    def get_ckpt_id(alg, run, epoch):
        return f'{alg}-{run}-{epoch}'

    def load_run(self, alg, run, dtype=np.float32):
        import h5py

        val_preds_local = {}
        with h5py.File(run + '/features/features.hdf5', "r") as f:
            epochs = f.keys()
            for epoch in epochs:
                ckpt_id = DomainNet126.get_ckpt_id(alg, run, epoch)
                if self.use_target_val:
                    tgt_preds = np.array(f[str(epoch)]['inference']['target_val']['logits'][:, :], dtype=dtype)  # shape (len(val), n_classes)
                else:
                    tgt_preds = np.array(f[str(epoch)]['inference']['target_train']['logits'][:, :], dtype=dtype)  # shape (len(val), n_classes)
                val_preds_local[ckpt_id] = tgt_preds

        return val_preds_local

    def load_runs(self, subsample_pct=100):
        """
        Load predictions into self.preds.
        """
        preds_dict = {}  # ckpt -> pred vector

        # All UDA algorithms tested
        algs = [d for d in os.listdir(self.location) if d != 'slurm' and not d.startswith('.')]

        # Prepare all (alg, run) pairs to process
        run_tasks = []

        for alg in algs:
            alg_runs = glob.glob(self.location + '/' + alg + '/[0-9]*')

            # Add all the (alg, run) pairs to the task list
            run_tasks.extend((alg, run) for run in alg_runs)

        # If subsample_pct is less than 100, subsample from the entire run_tasks list
        if subsample_pct < 100:
            n_subsample = max(1, int(len(run_tasks) * (subsample_pct / 100.0)))
            run_tasks = random.sample(run_tasks, n_subsample)

        # Split tasks into two lists: one for algs, and one for runs
        algs_list, runs_list = zip(*run_tasks)

        # Process all (alg, run) pairs concurrently
        with ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.load_run, algs_list, runs_list), total=len(run_tasks), desc=f'Processing runs'))

            # Collect predictions from each run
            for preds_local in results:
                preds_dict.update(preds_local)

        print("Loaded", len(preds_dict), "runs.")
        self.ckpts = list(preds_dict.keys())
        self.preds = np.array([preds_dict[k] for k in self.ckpts])

