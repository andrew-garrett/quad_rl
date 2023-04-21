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
        self.model = DynamicsNet(self.config)
        self.loss_fn = losses[self.loss_name]()
        self.metrics = {}
        self.step_outputs = {
            "train": [],
            "val": [],
            "test": []
        }
        self.nn_gt_min = torch.load(os.path.join(self.dataset_path, "train_nn_gt_min.pt"), map_location="cuda")
        self.nn_gt_max = torch.load(os.path.join(self.dataset_path, "train_nn_gt_max.pt"), map_location="cuda")
    
    def parse_config(self):
        self.batch_size = self.config["training"]["batch_size"]
        self.lr = self.config["training"]["lr"]
        self.n_drones = self.config["dataset"]["num_drones"]
        self.weight_decay = self.config["training"]["weight_decay"]
        self.optimizer_name = self.config["training"]["optimizer"]
        self.loss_name = self.config["training"]["loss_fn"]
        self.use_scheduler = self.config["training"]["scheduler"]
        self.dataset_path = os.path.join(self.config["dataset"]["root"], self.config["dataset"]["name"], "torch_dataset")
        self.output_size = self.config["model"]["accel_dim"]
    
    def configure_optimizers(self):
        optimizer = optimizers[self.optimizer_name](self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        opts = {"optimizer": optimizer}
        if self.use_scheduler:
            scheduler_config = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.25, patience=3, threshold=1e-4, threshold_mode="rel"),
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val/loss_epoch"
            }
            opts["lr_scheduler"] = scheduler_config
        return opts
    
    def step(self, batch, stage):
        state, action, acceleration = batch
        pred_accel = self.model(state.float(), action.float())
        loss = self.loss_fn(pred_accel, acceleration.float())
        pred_denorm, truth_denorm = (self.denormalize_accel(pred_accel.detach()), self.denormalize_accel(acceleration.detach()))
        errors = torch.abs(pred_denorm - truth_denorm)
        output_dict = {
            "loss": loss
        }
        for i in range(self.output_size):
            output_dict[f"error_{i+1}"] = errors[:,i]
        if stage == "train":
            self.log_metric(f"{stage}/loss", loss)
        self.step_outputs[stage].append(output_dict)
        return output_dict

    def denormalize_accel(self, accel):
        denormalized = (accel + 1)/2
        denormalized = (denormalized * (self.nn_gt_max - self.nn_gt_min)) + self.nn_gt_min
        return denormalized
    
    def training_step(self, batch, _):
        return self.step(batch, "train")
    
    def validation_step(self, batch, _):
        return self.step(batch, "val")
    
    def test_step(self, batch, _):
        return self.step(batch, "test")
    
    def epoch_end(self, stage):
        outputs = self.step_outputs[stage]
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log_metric(f"{stage}/loss_epoch", loss)
        for i in range(self.output_size):
            error_i = torch.cat([x[f"error_{i+1}"] for x in outputs]).mean()
            self.log_metric(f"{stage}/error_{i+1}_epoch", error_i)
        self.step_outputs[stage] = []
    
    def on_train_epoch_end(self):
        self.epoch_end("train")
    
    def on_validation_epoch_end(self):
        self.epoch_end("val")
    
    def on_test_epoch_end(self):
        self.epoch_end("test")
    
    def log_metric(self, metric_name, metric_value):
        if "epoch" in metric_name:
            self.log(metric_name, metric_value, on_epoch=True, on_step=False, prog_bar=True)
        else:
            self.log(metric_name, metric_value, on_epoch=False, on_step=True, prog_bar=True)

        if metric_name in self.metrics:
            self.metrics[metric_name].append(metric_value.item())
        else:
            self.metrics[metric_name] = [metric_value.item()]
    
    def save_metrics(self):
        filename = os.path.join(self.log_dir, "metrics.json")
        with open(filename, "w") as f:
            json.dump(self.metrics, f)
    
    def train_dataloader(self):
        train_dataset = DynamicsDataset("train", self.config)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
        return train_loader
    
    def val_dataloader(self):
        val_dataset = DynamicsDataset("val", self.config)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return val_loader
    
    def test_dataloader(self):
        test_dataset = DynamicsDataset("test", self.config)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        return test_loader