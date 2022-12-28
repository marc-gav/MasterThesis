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

# read only 50% of the dataset
df = pd.read_csv(
    "datasets/training_dataset_light.csv",
    sep=",",
    index_col=False,
    usecols=["word", "sentence_index", "cluster_label", "salience_value"],
)

ARCHITECTURE = input("Enter the architecture type: ")
FRAC = float(input("Enter the fraction of the dataset to use: [0, 1]: "))
df = df.sample(frac=FRAC, random_state=42)

dataset = ClusteredWordsDataset(df=df)
VOCAB_SIZE = dataset.get_vocab_size()
NUM_CLUSTERS = dataset.get_num_clusters()

TRAIN_SPLIT = float(input("Enter the train split: [0, 1]: "))
VAL_SPLIT = float(input("Enter the validation split: [0, 1]: "))
TRAIN_DATASET, VAL_DATASET = split_dataset(dataset, [TRAIN_SPLIT, VAL_SPLIT])

# normalize data


# Flatten the datapoints to fit them into the forward pass
TRAIN_DATASET.data = TRAIN_DATASET.data.view(TRAIN_DATASET.data.shape[0], -1)
VAL_DATASET.data = VAL_DATASET.data.view(VAL_DATASET.data.shape[0], -1)

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
    """Trains the model and logs the results to wandb"""
    run = wandb.init(
        config=RUN_CONFIG
    )  # config param gets ignored if its a sweep. Idk what to say,
    # it seems the shadiest implementation of an API I've ever seen
    wandb_logger = WandbLogger()

    # log architecture type
    wandb.run.summary["dataset_fraction"] = FRAC
    wandb.run.summary["architecture"] = ARCHITECTURE

    # log train and val splits
    wandb.run.summary["train_split"] = TRAIN_SPLIT
    wandb.run.summary["val_split"] = VAL_SPLIT

    model = ProbingClassifier(
        run.config,
        input_size=TRAIN_DATASET.data[0].nelement(),
        num_clusters=NUM_CLUSTERS,
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
    trainer = pl.Trainer(
        max_epochs=10000,
        logger=wandb_logger,
        log_every_n_steps=1,
        # accelerator="gpu",
        # devices=1,
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
