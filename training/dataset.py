import os
import torch
from torch.utils.data import Dataset

class DynamicsDataset(Dataset):
    def __init__(self, stage, config) -> None:
        super().__init__()
        self.config = config
        self.dataset_path = os.path.join("bootstrap/datasets", self.config["dataset"]["name"])
        self.dataset_path = os.path.join(self.dataset_path, "torch_dataset")
        self.states = torch.load(os.path.join(self.dataset_path, f"{stage}_states.pt"))
        self.actions = torch.load(os.path.join(self.dataset_path, f"{stage}_actions.pt"))
        self.targets = torch.load(os.path.join(self.dataset_path, f"{stage}_targets.pt"))
    
    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, index):
        state = self.states[index]
        action = self.actions[index]
        acceleration = self.targets[index]
        return state, action, acceleration