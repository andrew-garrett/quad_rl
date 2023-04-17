import numpy as np
import os
import json
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from training.lightning import DynamicsLightningModule


def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        default="training_config.json",
        help="path_to_config_file",
        type=str
    )
    parser.add_argument(
        "--epochs",
        default=50,
        help="Number of training epochs",
        type=int
    )
    args = parser.parse_args()
    return args

def build_trainer_module(config, experiment_dir, epochs):
    log_dir = f"{experiment_dir}/logs"
    weights_dir = f"{experiment_dir}/models"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    wandb_logger = WandbLogger(project="ESE650 Final Project", log_model="all") #, group=config["dataset"]["name"])
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        # dirpath=weights_dir,
        # filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        default_root_dir=weights_dir,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    module = DynamicsLightningModule(config, log_dir)
    return trainer, module

def create_experiment_dir():
    os.makedirs("training_logs", exist_ok=True)
    runs = os.listdir("training_logs")
    if len(runs) == 0:
        run = 0
    else:
        run = max([int(s.replace('run_', '')) for s in runs]) + 1
    experiment_dir = f"training_logs/run_{run}"
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config_path)
    experiment_dir = create_experiment_dir()
    trainer, module = build_trainer_module(config, experiment_dir, args.epochs)
    trainer.fit(module)
    # module.save_metrics()
    # module.render_videos()