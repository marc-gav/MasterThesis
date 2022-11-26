import pandas as pd
import wandb
import yaml
import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.wandb import WandbLogger

from bertologist.data.Models import ProbingClassifier
from bertologist.data.Datasets import ClusteredWordsDataset
from bertologist.data.utils import split_dataset

with open("experiments/config.yaml") as f:
    RUN_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

with open("experiments/sweep.yaml") as f:
    SWEEP_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

df = pd.read_csv(
    "datasets/training_dataset_light.csv", sep=",", index_col=False
)
dataset = ClusteredWordsDataset(df=df)
VOCAB_SIZE = dataset.get_vocab_size()
NUM_CLUSTERS = dataset.get_num_clusters()

TRAIN_DATASET, VAL_DATASET = split_dataset(dataset, [0.8, 0.2])

# Flatten the datapoints to fit them into the forward pass
TRAIN_DATASET.data = TRAIN_DATASET.data.view(TRAIN_DATASET.data.shape[0], -1)
VAL_DATASET.data = VAL_DATASET.data.view(VAL_DATASET.data.shape[0], -1)

# Fix the seeds
SEED = 42
pl.seed_everything(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


def train_experiment():
    """Trains the model and logs the results to wandb"""
    run = wandb.init(
        config=RUN_CONFIG
    )  # config param gets ignored if its a sweep. Idk what to say,
    # it seems the shadiest implementation of an API I've ever seen
    wandb_logger = WandbLogger()
    model = ProbingClassifier(
        run.config, vocab_size=VOCAB_SIZE, num_clusters=NUM_CLUSTERS
    )

    wandb.watch(model, log="all", log_freq=50, log_graph=True)
    train_dataloader = DataLoader(
        TRAIN_DATASET,
        batch_size=run.config["batch_size"],
        shuffle=False,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        VAL_DATASET,
        batch_size=run.config["batch_size"],
        shuffle=False,
        num_workers=4,
    )
    trainer = pl.Trainer(
        max_epochs=10000,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_loss", patience=3, mode="min"
            )
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    wandb.finish()


def sweep():
    """Runs a sweep on the model"""

    project_name = input("Enter the wandb project name: ")
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project_name)
    wandb.agent(
        sweep_id,
        function=train_experiment,
    )


if __name__ == "__main__":
    train_experiment()
