import pandas as pd
import wandb
import yaml
import pytorch_lightning as pl
import torch
import numpy as np
import gzip
import os
import pickle
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.wandb import WandbLogger

from bertologist.data.Models import ProbingClassifier
from bertologist.data.Datasets import (
    ClusteredWordsDataset,
    ClusteredBOWDataset,
)
from bertologist.data.utils import split_dataset

with open("experiments/config.yaml") as f:
    RUN_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

with open("experiments/sweep.yaml") as f:
    SWEEP_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

with gzip.open("datasets/light_training_dataset_bow.pkl.zip", "rb") as f:
    bow_dataset = pickle.load(f)

ARCHITECTURE = input("Enter the architecture type: ")
NOTES = input("Enter any notes: ")

VOCAB_SIZE = bow_dataset.get_vocab_size()
NUM_CLUSTERS = bow_dataset.get_num_clusters()

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2
TRAIN_DATASET, VAL_DATASET = split_dataset(
    bow_dataset, [TRAIN_SPLIT, VAL_SPLIT]
)

# class frequency of TRAIN_DATASET
CLASS_FREQ = TRAIN_DATASET.get_class_freq()

print(
    f"Number of datapoints: {len(TRAIN_DATASET.data)}",
    f"Size of each datapoint: {TRAIN_DATASET.data[0].nelement() * TRAIN_DATASET.data[0].element_size() / 1e6} MB",
    f"Size of the whole dataset: {len(TRAIN_DATASET.data) * TRAIN_DATASET.data[0].nelement() * TRAIN_DATASET.data[0].element_size() / 1e9} GB",
    sep="\n",
)

# Fix the seeds
SEED = 42
pl.seed_everything(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def train_experiment():

    # wj=n_samples / (n_classes * n_samplesj)
    class_weights = torch.sum(CLASS_FREQ) / (NUM_CLUSTERS * CLASS_FREQ)

    # send class weights to the GPU
    if torch.cuda.is_available():
        ACCELERATOR = "gpu"
        class_weights = class_weights.cuda()
    else:
        # use SparseCsrCPU backend
        ACCELERATOR = "cpu"

    """Trains the model and logs the results to wandb"""
    run = wandb.init(
        config=RUN_CONFIG
    )  # config param gets ignored if its a sweep. Idk what to say,
    # it seems the shadiest implementation of an API I've ever seen
    wandb_logger = WandbLogger()

    wandb.run.summary["architecture"] = ARCHITECTURE

    # log train and val splits
    wandb.run.summary["train_split"] = TRAIN_SPLIT
    wandb.run.summary["val_split"] = VAL_SPLIT
    wandb.run.summary["notes"] = NOTES

    model = ProbingClassifier(
        run.config,
        input_size=TRAIN_DATASET.data[0].nelement(),
        num_clusters=NUM_CLUSTERS,
        class_weights=class_weights,
    )

    # print infomration about TRAIN_DATASET:
    # number of datapoints
    # size in memory (in MB) of each datapoint
    # size in memory (in GB) of the whole dataset

    # wandb.watch(model, log="all", log_freq=50, log_graph=True)

    train_dataloader = DataLoader(
        TRAIN_DATASET,
        batch_size=run.config["batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        VAL_DATASET,
        batch_size=run.config["batch_size"],
        shuffle=False,
    )

    # log flops
    trainer = pl.Trainer(
        max_epochs=10000,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator=ACCELERATOR,
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, min_delta=0.01
            )
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # save model architecture
    wandb.save(os.path.join(wandb.run.dir, "bertologist/data/Models.py"))
    wandb.finish()


def sweep():
    """Runs a sweep on the model"""

    sweep_id = wandb.sweep(SWEEP_CONFIG, project="Master Thesis")

    wandb.agent(
        sweep_id,
        function=train_experiment,
    )


if __name__ == "__main__":
    sweep()
