from datasets import DomainNet126
import numpy as np

class Oracle:
    
    def __init__(self, dataset, base_dir="../powerful-benchmarker/datasets/"):
        if isinstance(dataset, DomainNet126):
            test_set = dataset.task.split("_")[-1]
            if dataset.use_target_val:
                test_set_txt = f'{base_dir}/domainnet/{test_set}126_test.txt'
            else:
                test_set_txt = f'{base_dir}/domainnet/{test_set}126_train.txt'
        else:
            raise NotImplementedError()
        
        with open(test_set_txt, 'r') as f:
            self.labels = np.array([ int(s.split(" ")[-1].replace("\n","")) for s in f.readlines()])

    def __call__(self, idx):
        return self.labels[idx]

class User:
    # not implemented
    pass