import os
import torch
from torch.utils.data import Dataset

class DynamicsDataset(Dataset):
    def __init__(self, stage, config) -> None:
        super().__init__()
        self.config = config
        self.dataset_path = os.path.join(self.config["dataset"]["root"], self.config["dataset"]["name"], "torch_dataset")
        self.states = torch.load(os.path.join(self.dataset_path, f"{stage}_states.pt"))
        self.control_targets = torch.load(os.path.join(self.dataset_path, f"{stage}_control_targets.pt"))
        self.nn_gts = torch.load(os.path.join(self.dataset_path, f"{stage}_nn_gts.pt"))
    
    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, index):
        state = self.states[index]
        control_target = self.control_targets[index]
        acceleration = self.nn_gts[index]
        return state, control_target, acceleration