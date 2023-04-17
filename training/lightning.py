import os
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .nn import DynamicsNet
from .dataset import DynamicsDataset

optimizers = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop
}

losses = {
    "mse": torch.nn.MSELoss,
    "mae": torch.nn.L1Loss,
}

class DynamicsLightningModule(pl.LightningModule):
    def __init__(self, config, log_dir):
        super().__init__()
        self.config = config
        self.log_dir = log_dir
        self.parse_config()
        self.save_hyperparameters()
        self.model = DynamicsNet(self.config)
        self.loss_fn = losses[self.loss_name]()
        self.metrics = {}
        self.step_outputs = {
            "train": [],
            "val": [],
            "test": []
        }
    
    def parse_config(self):
        self.batch_size = self.config["training"]["batch_size"]
        self.lr = self.config["training"]["lr"]
        self.n_drones = self.config["dataset"]["num_drones"]
        self.weight_decay = self.config["training"]["weight_decay"]
        self.optimizer_name = self.config["training"]["optimizer"]
        self.loss_name = self.config["training"]["loss_fn"]
    
    def configure_optimizers(self):
        return optimizers[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    
    def step(self, batch, stage):
        state, action, acceleration = batch
        pred_accel = self.model(state.float(), action.float())
        loss = self.loss_fn(pred_accel, acceleration.float())
        output_dict = {
            "loss": loss
        }
        self.step_outputs[stage].append(output_dict)
        return output_dict
    
    def training_step(self, batch, _):
        return self.step(batch, "train")
    
    def validation_step(self, batch, _):
        return self.step(batch, "val")
    
    def test_step(self, batch, _):
        return self.step(batch, "test")
    
    def epoch_end(self, stage):
        outputs = self.step_outputs[stage]
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log_metric(f"{stage}/loss", loss)
        # self.log(f"{stage}/loss", loss, on_epoch=True)
        self.step_outputs[stage] = []
    
    def on_train_epoch_end(self):
        self.epoch_end("train")
    
    def on_validation_epoch_end(self):
        self.epoch_end("val")
    
    def on_test_epoch_end(self):
        self.epoch_end("test")
    
    def log_metric(self, metric_name, metric_value):
        self.log(metric_name, metric_value, on_epoch=True, prog_bar=True)
        if metric_name in self.metrics:
            self.metrics[metric_name].append(metric_value.item())
        else:
            self.metrics[metric_name] = [metric_value.item()]
    
    # def save_metrics(self):
    #     filename = os.path.join(self.log_dir, "metrics.json")
    #     with open(filename, "w") as f:
    #         json.dump(self.metrics, f)
    
    def train_dataloader(self):
        train_dataset = DynamicsDataset("train", self.config)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,num_workers=8)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = DynamicsDataset("val", self.config)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return val_loader
    
    def test_dataloader(self):
        test_dataset = DynamicsDataset("test", self.config)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return test_loader