import numpy as np
import os
import json
import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from training.lightning import DynamicsLightningModule


def load_config(config_path, root, dataset_name):
    with open(config_path) as f:
        config = json.load(f)
        config["dataset"]["root"] = root
        config["dataset"]["name"] = dataset_name
    return config

def load_sweep_config(config, sweep_config_path):
    with open(sweep_config_path) as f:
        sweep_config = json.load(f)
    for k, v in sweep_config.items():
        split_k = k.split("_")
        config[split_k[0]]["_".join(split_k[1:])] = v
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Dynamics Neural Network Training Script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--root",
        default="./bootstrap/datasets/testing/dyn/",
        help="root where datasets are stored",
        type=str
    )
    parser.add_argument(
        "--dataset",
        default="AGGRO_000",
        help="name of the dataset",
        type=str
    )
    parser.add_argument(
        "--config",
        default="configs/training_config.json",
        help="path_to_config_file",
        type=str
    )
    parser.add_argument(
        "--epochs",
        default=50,
        help="Number of training epochs",
        type=int
    )
    parser.add_argument(
        "--sweep",
        help="sweep config file",
        type=str
    )
    args = parser.parse_args()
    return args

def build_trainer_module(config, experiment_dir, epochs):
    log_dir = os.path.join(experiment_dir, "logs")
    weights_dir = os.path.join(experiment_dir, "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    logging_config = {
        "reinit": True,
        "project": "ESE650 Final Project",
        "group": config["dataset"]["name"]
    }

    wandb_config = {}
    for k, v in config.items():
        for sub_k, sub_v in v.items():
            wandb_config[f"{k}_{sub_k}"] = sub_v
    run = wandb.init(config=wandb_config, **logging_config)
    wandb_logger = WandbLogger(log_model="all", experiment=run)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss_epoch",
        mode="min",
        # dirpath=weights_dir,
        # filename="model-{epoch:02d}-{val_loss:.2f}",
    )
    try:
        trainer = pl.Trainer(
            devices=1,
            accelerator="gpu",
            max_epochs=epochs,
            log_every_n_steps=50,
            # default_root_dir=weights_dir,
            logger=wandb_logger,
            callbacks=[checkpoint_callback,
                       LearningRateMonitor(logging_interval='epoch')
                      ]
        )
    except:
        trainer = pl.Trainer(
            devices=1,
            accelerator="cpu",
            max_epochs=epochs,
            log_every_n_steps=50,
            # default_root_dir=weights_dir,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
        )
    module = DynamicsLightningModule(config, log_dir)
    return trainer, module

def create_experiment_dir(config):
    training_logs_dir = os.path.join(config["dataset"]["root"], config["dataset"]["name"], "training_logs")
    os.makedirs(training_logs_dir, exist_ok=True)
    runs = os.listdir(training_logs_dir)
    if len(runs) == 0:
        run = 0
    else:
        run = max([int(s.replace('run_', '')) for s in runs]) + 1
    experiment_dir = os.path.join(training_logs_dir, f"run_{run}")
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config, args.root, args.dataset)
    if args.sweep is not None:
        config = load_sweep_config(config, args.sweep)
    experiment_dir = create_experiment_dir(config)
    trainer, module = build_trainer_module(config, experiment_dir, args.epochs)
    trainer.fit(module)
    wandb.finish()
    # module.save_metrics()
    # module.render_videos()