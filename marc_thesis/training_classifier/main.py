import wandb
import yaml
import pytorch_lightning as pl
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from pytorch_lightning.loggers.wandb import WandbLogger
from sklearn.model_selection import train_test_split
import re
from loguru import logger
import random
import pickle

from marc_thesis.training_classifier.Models import (
    ProbingClassifierLinear,
    ProbingClassifierTwoLayer,
    ProbingClassifierThreeLayer,
)
from marc_thesis.training_classifier.TorchDatasets import (
    preprocess_data,
    preprocess_labels,
    prepare_for_train_val,
    ClusteredBOWDataset,
)
from marc_thesis.utils.file_manager import load_file_target

# create a named tuple for the different datasets
from collections import namedtuple

DatasetProperties = namedtuple(
    "DatasetProperties",
    [
        "salience_method",
        "cluster_method",
        "num_clusters",
        "is_salience_weighting",
    ],
)

USER_SETTINGS = {}
EXPERIMENTS_FOLDER = os.environ.get("MARC_THESIS_EXPERIMENT_FOLDER")
SEED = 42
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.2

pl.seed_everything(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# get file directory
file_dir = os.path.dirname(os.path.realpath(__file__))

with open(f"{file_dir}/config.yaml") as f:
    RUN_CONFIG = yaml.load(f, Loader=yaml.FullLoader)

with open(f"{file_dir}/sweep.yaml") as f:
    SWEEP_CONFIG = yaml.load(f, Loader=yaml.FullLoader)


def get_dataset_class_weights(
    dataset: ClusteredBOWDataset, method: str = "uniform"
):
    """Returns the weights that should be used for each class in the dataset in order to
    accout for the class imbalance."""

    n_classes = torch.unique(dataset.labels).shape[0]
    n_samples_per_class = torch.bincount(dataset.labels)

    if method == "INS":  # inverse number of samples
        return 1.0 / n_samples_per_class.float()
    elif method == "ISNS":  # inverse square root of number of samples
        return 1.0 / torch.sqrt(n_samples_per_class.float())
    elif method == "ENS":  # effective number of samples
        beta = 0.99
        effective_num = 1.0 - torch.pow(beta, n_samples_per_class)
        weights_for_samples = (1.0 - beta) / torch.tensor(effective_num)
        weights_for_samples = (
            weights_for_samples / torch.sum(weights_for_samples) * n_classes
        )
        return weights_for_samples
    elif method == "uniform":
        return torch.ones(n_classes)
    else:
        raise ValueError(f"Unrecognized method: {method}")


def log_dataset_info(train_val_data: dict):
    """Logs the dataset info to wandb"""
    # First log the different keys in the dict
    logger.info("The following are the differnt datasets ready to train on:")
    for key, data_dict in train_val_data.items():
        train_data = data_dict["train_data"]

        logger.info(
            f"Salience score: {key[0]} | Clustering method: {key[1]} | Number of clusters: {key[2]}"
        )
        logger.info(f"\\---Number of datapoints: {len(train_data.data)}")
        logger.info(
            f"\\--Size of each datapoint: {train_data.data[0].nelement() * train_data.data[0].element_size() / 1e6} MB\n"
        )
        logger.info(
            f"\\--Size of the whole train_data: {len(train_data.data) * train_data.data[0].nelement() * train_data.data[0].element_size() / 1e9} GB"
        )


def train_experiment(RUN_CONFIG: dict = {}):
    """Trains the model and logs the results to wandb"""
    run = wandb.init(
        config=RUN_CONFIG
    )  # config param gets ignored if its a sweep. Idk what to say,
    # it seems the shadiest implementation of an API I've ever seen

    accelerator = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_logger = WandbLogger()
    dataset_key = tuple(run.config["dataset_key"])
    (
        salience_method,
        cluster_method,
        num_clusters,
        is_salience_weighting,
    ) = dataset_key
    run.summary["salience_method"] = salience_method
    run.summary["cluster_method"] = cluster_method
    run.summary["num_clusters"] = num_clusters
    run.summary["is_salience_weighting"] = is_salience_weighting

    architecture = run.config["architecture"]
    train_dataset = TRAIN_VAL_DATASETS[dataset_key]["train_data"]
    val_dataset = TRAIN_VAL_DATASETS[dataset_key]["val_data"]
    class_defrequencing_method = run.config["class_defrequencing_method"]

    num_clusters = int(torch.max(train_dataset.labels) + 1)
    class_weights = get_dataset_class_weights(
        train_dataset, method=class_defrequencing_method
    )

    # send class_weights to accelerator
    class_weights = class_weights.to(accelerator)

    assert (
        class_weights.shape[0] == num_clusters
    ), f"class_weights shape is {class_weights.shape} but num_clusters is {num_clusters}"

    if architecture == "linear":
        ProbingClassifier = ProbingClassifierLinear
    elif architecture == "two_layer":
        ProbingClassifier = ProbingClassifierTwoLayer
    elif architecture == "three_layer":
        ProbingClassifier = ProbingClassifierThreeLayer
    else:
        raise NotImplementedError(f"{architecture} is not implemented")

    model = ProbingClassifier(
        run.config,
        input_size=train_dataset.get_saliences().shape[1],
        num_clusters=num_clusters,
        class_weights=class_weights,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=run.config["batch_size"],
        shuffle=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=run.config["batch_size"],
        shuffle=False,
    )

    trainer = pl.Trainer(
        max_epochs=10000,
        logger=wandb_logger,
        log_every_n_steps=1,
        accelerator=accelerator,
        devices=1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_f1", patience=8, min_delta=0.01
            )
        ],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    wandb.finish()


def debug_run(
    target_word: str,
    train_val_data: dict,
):
    dataset_key = list(train_val_data.keys())[0]
    run_config = {
        run_key: random.choice(SWEEP_CONFIG["parameters"][run_key]["values"])
        for run_key in SWEEP_CONFIG["parameters"]
        if run_key != "learning_rate"
    }
    run_config["learning_rate"] = 0.01
    run_config["dataset_key"] = dataset_key

    train_experiment(run_config)


def run_sweeps(
    target_word: str,
    train_val_data: dict,
):
    """Runs a sweep on the model"""

    for key in train_val_data.keys():
        SWEEP_CONFIG["parameters"]["dataset_key"] = {}
        SWEEP_CONFIG["parameters"]["dataset_key"]["values"] = [key]
        sweep_id = wandb.sweep(
            SWEEP_CONFIG, project=f"Master Thesis_{target_word}_context_window_3"
        )

        wandb.agent(
            sweep_id,
            function=train_experiment,
            count=100,
        )

        logger.info(f"Sweep: {sweep_id} with config: {SWEEP_CONFIG} finished")


def get_salience_datasets(target_word: str) -> dict:
    experiment_files_dir = os.path.join(EXPERIMENTS_FOLDER, target_word)
    target_word_files = os.listdir(experiment_files_dir)
    salience_datasets = {}
    # Find all of the files that end in _salience_scores.pickle
    salience_pattern = re.compile(f".*_salience_scores.pickle")
    for file in target_word_files:
        if re.search(salience_pattern, file):
            salience_method = file.split("_")[1]
            salience_file = os.path.join(experiment_files_dir, file)
            salience_datasets[salience_method] = pickle.load(
                open(salience_file, "rb")
            )

    return salience_datasets


def get_cluster_datasets(target_word: str) -> dict:
    experiment_files_dir = os.path.join(EXPERIMENTS_FOLDER, target_word)
    target_word_files = os.listdir(experiment_files_dir)
    cluster_datasets = {}
    cluster_pattern = re.compile(f"{target_word}_cluster_labels_")
    if any([re.search(cluster_pattern, file) for file in target_word_files]):
        for file in target_word_files:
            if not file.endswith(".pickle"):
                continue
            file = file.split(".")[0]
            if re.search(cluster_pattern, file):
                num_clusters = int(file.split("_")[-2])
                cluster_method = file.split("_")[-3]
                cluster_datasets[
                    (cluster_method, num_clusters)
                ] = load_file_target(target_word, f"{file}.pickle")

    return cluster_datasets


def get_train_val_datasets(
    salience_datasets: dict, cluster_datasets: dict
) -> dict:
    train_val_data = {}
    # Combine the keys of the salience and cluster datasets
    for salience_method in salience_datasets.keys():
        for cluster_method, num_clusters in cluster_datasets.keys():
            for salience_weighting in [True, False]:
                (
                    sentence_idx_to_salience,
                    sentence_idx_to_words,
                ) = preprocess_data(salience_datasets[salience_method])

                sentence_idx_to_labels = preprocess_labels(
                    cluster_datasets[(cluster_method, num_clusters)]
                )

                data, labels = prepare_for_train_val(
                    sentence_idx_to_salience,
                    sentence_idx_to_words,
                    sentence_idx_to_labels,
                    is_salience_weighting=salience_weighting,
                )

                (
                    train_data,
                    val_data,
                    train_labels,
                    val_labels,
                ) = train_test_split(
                    data,
                    labels,
                    train_size=TRAIN_SPLIT,
                    test_size=VAL_SPLIT,
                    random_state=SEED,
                )

                train_data = ClusteredBOWDataset(
                    saliences=train_data, labels=train_labels
                )

                val_data = ClusteredBOWDataset(
                    saliences=val_data, labels=val_labels
                )

                defrequency_weights = get_dataset_class_weights(train_data)

                train_val_data[
                    DatasetProperties(
                        salience_method=salience_method,
                        cluster_method=cluster_method,
                        num_clusters=num_clusters,
                        is_salience_weighting=salience_weighting,
                    )
                ] = {
                    "train_data": train_data,
                    "val_data": val_data,
                    "class_weights": defrequency_weights,
                }

    return train_val_data


def main():
    target_word = input("Enter the target word: ")

    salience_datasets = get_salience_datasets(target_word)
    # modify salience_datasets to keep the first 3 elements
    for salience_method in salience_datasets.keys():
        dicts = salience_datasets[salience_method]
        dicts = {
            key: {
                k: v[:3]
                for k, v in value.items()
            }
            for key, value in dicts.items()
        }
        salience_datasets[salience_method] = dicts

    cluster_datasets = get_cluster_datasets(target_word)

    train_val_data = get_train_val_datasets(
        salience_datasets, cluster_datasets
    )

    # Make train_val_data public
    global TRAIN_VAL_DATASETS
    TRAIN_VAL_DATASETS = train_val_data

    # train_config = yaml.load("config.yaml", Loader=yaml.FullLoader)
    # train_experiment(train_config)
    run_sweeps(target_word, train_val_data)


if __name__ == "__main__":
    main()
