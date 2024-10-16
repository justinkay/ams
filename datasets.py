import os
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import random
import torch
import h5py
import logging


class DomainNet126:
    """DomainNet126. 
    Assumes experiments were run with powerful-benchmarker and follow the directory structure defined there.
    """

    def __init__(self, task, base_dir="../powerful-benchmarker/experiments/", use_target_val=False, 
                 dat_filename="predictions.pt", ckpt_filename="ckpts.txt"):
        """
        Args:
            task: should be {source}_{target}, e.g. "sketch_painting"
            use_target_val: Whether to use the target validation set. Use the target training set if False. False by default for
                            powerful-benchmarker experiments (i.e. those written to validator_ScoreHistory.csv)
            dat_filename: Path to the .dat file where predictions will be saved incrementally.
            ckpt_filename: Path to the file where checkpoint IDs will be stored in order.
        """
        self.location = os.path.join(base_dir, f"domainnet126_{task}_fl6_Adam_lr1")
        self.use_target_val = use_target_val
        self.task = task

        self.dat_file = os.path.join(self.location, dat_filename)
        self.ckpt_file = os.path.join(self.location, ckpt_filename)
        self.ckpts = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_ckpt_id(alg, run, epoch):
        return f'{alg}-{run}-{epoch}'

    def load_run(self, alg, run, dtype=torch.float32):
        val_preds_local = {}
        hdf5_file_path = os.path.join(run, 'features/features.hdf5')

        try:
            with h5py.File(hdf5_file_path, "r") as f:
                epochs = f.keys()
                for epoch in epochs:
                    ckpt_id = DomainNet126.get_ckpt_id(alg, run, epoch)
                    if self.use_target_val:
                        tgt_preds = torch.tensor(f[str(epoch)]['inference']['target_val']['logits'][:, :], dtype=dtype)
                    else:
                        tgt_preds = torch.tensor(f[str(epoch)]['inference']['target_train']['logits'][:, :], dtype=dtype)
                    val_preds_local[ckpt_id] = tgt_preds

        except FileNotFoundError as fnfe:
            logging.error(fnfe)
            return {}
        except Exception as e:
            logging.error(f"An error occurred while processing {hdf5_file_path}: {e}")
            return {}

        return val_preds_local


    def load_runs(self, subsample_pct=100, batch_size=100, force_reload=False):
        """
        Load predictions into self.preds and save them incrementally to a .dat file.
        Args:
            force_reload: If True, re-generate .dat and checkpoint files. If False, attempt to load existing files.
        """
        # Check if the .dat and ckpt files already exist and load them if force_reload is False
        if not force_reload and os.path.exists(self.dat_file) and os.path.exists(self.ckpt_file):
            print("Found existing files. Loading from disk.")
            self.load_ckpts()
            self.load_preds()
            print(f"Loaded {len(self.ckpts)} checkpoints from {self.ckpt_file}")
            return

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
        ckpts_per_run = 20 # TODO get this dynamically

        # Initialize tensor to save to the file
        total_runs = len(run_tasks)
        run_shape = (total_runs*ckpts_per_run, 7126, 126)
        self.preds = torch.empty(run_shape, dtype=torch.float32, device=self.device)

        print(f"Loading predictions from {total_runs} runs from HDF5 files...")

        with open(self.ckpt_file, 'w') as ckpt_file:
            current_index = 0
            for i in tqdm(range(0, len(run_tasks), batch_size), desc='Processing batches'):
                batch = run_tasks[i:i+batch_size]
                results = [self.load_run(alg, run) for alg, run in batch]
                
                for preds_local in results:
                    for ckpt_id, preds in preds_local.items():
                        self.preds[current_index] = preds.to(self.device)
                        ckpt_file.write(f"{ckpt_id}\n")
                        self.ckpts.append(ckpt_id)
                        current_index += 1

        print(f"Saved {len(self.ckpts)} checkpoints to {self.ckpt_file}")
        torch.save(self.preds, self.dat_file)

    def load_ckpts(self):
        """
        Load the checkpoint IDs from the saved ckpt_file.
        """
        with open(self.ckpt_file, 'r') as ckpt_file:
            self.ckpts = [line.strip() for line in ckpt_file]
        return self.ckpts

    def load_preds(self):
        self.preds = torch.load(self.dat_file, map_location=self.device)
        return self.preds
