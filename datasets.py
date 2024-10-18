import os
import glob
import numpy as np
from tqdm import tqdm
import time
import torch
import h5py
import logging
import contextlib
import random


class DomainNet126:
    """DomainNet126. 
    Assumes experiments were run with powerful-benchmarker and follow the directory structure defined there.
    """

    def __init__(self, task, base_dir="../powerful-benchmarker/experiments/", use_target_val=False, 
                 dat_filename="predictions.pt", ckpt_filename="ckpts.txt", dataset_filter=None):
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

        self.dataset_filter = dataset_filter
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

    def get_run_tasks(self):
        # All UDA algorithms tested
        algs = [d for d in os.listdir(self.location) if d != 'slurm' and not d.startswith('.')]

        if self.dataset_filter is not None:
            algs = self.dataset_filter.split(",")
            print("Filtering UDA algorithms down to:", algs)

        # Prepare all (alg, run) pairs to process
        run_tasks = []

        for alg in algs:
            alg_runs = glob.glob(self.location + '/' + alg + '/[0-9]*')

            # Add all the (alg, run) pairs to the task list
            run_tasks.extend((alg, run) for run in alg_runs)

        return run_tasks

    def get_total_run_shape(self, run_tasks):
        """
        Determine the total run shape by iterating through the provided (subsampled) run tasks.
        Args:
            run_tasks: List of (alg, run) pairs that represent the subsampled runs.
        """
        total_checkpoints = 0
        samples = 0
        classes = 0

        for alg, run in tqdm(run_tasks, desc="Calculating total run shape"):
            hdf5_file_path = os.path.join(run, 'features/features.hdf5')
            try:
                with h5py.File(hdf5_file_path, "r") as f:
                    epochs = f.keys()
                    total_checkpoints += len(epochs)

                    # Get samples and classes from the first epoch
                    first_epoch = next(iter(epochs))
                    if self.use_target_val:
                        logits = f[first_epoch]['inference']['target_val']['logits']
                    else:
                        logits = f[first_epoch]['inference']['target_train']['logits']

                    if samples == 0:  # Only set these once
                        samples, classes = logits.shape

            except Exception as e:
                logging.warning(f"Couldn't read {hdf5_file_path}: {e}")
                continue

        if total_checkpoints == 0 or samples == 0 or classes == 0:
            logging.error("Failed to determine run shape. Using default values.")
            return (1000, 7126, 126)  # Default values

        return (total_checkpoints, samples, classes)

    def load_runs(self, subsample_pct=100, batch_size=2, force_reload=False, num_workers=8, write=True):
        if subsample_pct < 100 or self.dataset_filter is not None:
            print("Disabling write and enabling force_reload because we are subsampling.")
            write = False
            force_reload = True
        
        if not force_reload and os.path.exists(self.dat_file) and os.path.exists(self.ckpt_file):
            print("Found existing files. Loading from disk.")
            start_time = time.time()
            self.load_ckpts()
            self.load_preds()
            elapsed_time = time.time() - start_time
            readable_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            print(f"Loaded {len(self.ckpts)} checkpoints from {self.ckpt_file} in {readable_time}")
            return

        run_tasks = self.get_run_tasks()
        if subsample_pct < 100:
            num_samples = int(len(run_tasks) * (subsample_pct / 100))
            run_tasks = random.sample(run_tasks, num_samples)
            print(f"Subsampled {num_samples} out of {len(run_tasks)} total runs ({subsample_pct}%)")

        run_shape = self.get_total_run_shape(run_tasks)
        print(f"Total run shape: {run_shape}")

        self.preds = torch.empty(run_shape, dtype=torch.float32, device=self.device)
        print(f"Loading predictions from {len(run_tasks)} runs from HDF5 files...")

        with open(self.ckpt_file, 'w') if write else contextlib.nullcontext() as ckpt_file:
            current_index = 0
            for task in tqdm(run_tasks):
                preds_local = self.load_run(*task)
                for ckpt_id, preds in preds_local.items():
                    self.preds[current_index] = preds.to(self.device)
                    self.ckpts.append(ckpt_id)
                    current_index += 1
                    if write:
                        ckpt_file.write(f"{ckpt_id}\n")

        if write:
            print(f"Saved {len(self.ckpts)} checkpoints to {self.ckpt_file}")
            print("Saving .pt file...")
            torch.save(self.preds, self.dat_file, pickle_protocol=4) # pickle protocol helps with OOM
            print("Saved.")

    def load_ckpts(self):
        """
        Load the checkpoint IDs from the saved ckpt_file.
        """
        with open(self.ckpt_file, 'r') as ckpt_file:
            self.ckpts = [line.strip() for line in ckpt_file]
        return self.ckpts

    def load_preds(self):
        self.preds = torch.load(self.dat_file, map_location=self.device, weights_only=False)
        return self.preds
