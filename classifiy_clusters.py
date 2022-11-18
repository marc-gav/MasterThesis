import math
import pandas as pd
from torch import optim, nn, utils, Tensor
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_config import Config


class ClusteredWordsDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, data: torch.Tensor = None, labels: torch.Tensor = None):
        assert (df is None) != (data is None and labels is None), "Must provide either df or data and labels"
        if data is not None and labels is not None:
            self.data = data
            self.labels = labels
        else:
            df = df
            word_to_idx: dict = {}
            for word in df["word"]:
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)
            vocab_size = len(word_to_idx)

            self.sentences = df["sentence_index"].unique()
            self.data = torch.zeros((len(self.sentences), 10, vocab_size))
            self.labels = torch.zeros((len(self.sentences)), dtype=torch.long)

            # We use two different indices because we don't trust the data.
            # The index that comes from the dataframe 
            # is not guaranteed to be in order or even start at 0, it only provides us with
            # information to group words into the same sentence
            for sentence_num, sentence_idx in enumerate(self.sentences): 
                sentence_df = df[df["sentence_index"] == sentence_idx]
                words = sentence_df["word"].values
                cluster_value = sentence_df["cluster_label"].values[0]
                
                one_hot_matrix = torch.zeros((10, vocab_size)) # Fixed size 10. If the sentence is shorter, the rest is implicitly padded with 0s
                for i, word in enumerate(words):
                    attention = sentence_df["attention"].values[i]
                    one_hot_matrix[i, word_to_idx[word]] = attention
                self.data[sentence_num, :, :] = one_hot_matrix
                self.labels[sentence_num] = cluster_value

            unique_labels = torch.unique(self.labels)
            assert torch.all(torch.sort(unique_labels)[0] == torch.arange(unique_labels.max() + 1)), "Labels must be consecutive integers starting at 0"
            # One-hot encode the labels
            one_hot_labels = torch.zeros((len(self.labels), 10))
            for i, label in enumerate(self.labels):
                one_hot_labels[i, label] = 1
            self.labels = one_hot_labels          

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def split_dataset(dataset: ClusteredWordsDataset, split_values: list):
    """Splits the dataset into multiple datasets based on the split_values.
    The split_values should be a list of floats that sum to 1.
    """

    assert sum(split_values) == 1, "The split values should sum to 1"
    assert len(split_values) > 1, "There should be at least 2 split values"

    dataset_sizes = [int(len(dataset) * split_value) for split_value in split_values]
    leftover = len(dataset) - sum(dataset_sizes)
    dataset_sizes[0] += leftover

    # Split dataset.data accordingly
    data_splits = torch.split(dataset.data, dataset_sizes)
    label_splits = torch.split(dataset.labels, dataset_sizes)
    datasets = []
    for data, labels in zip(data_splits, label_splits):
        datasets.append(ClusteredWordsDataset(data=data, labels=labels))
    
    return datasets
        

class ProbingClassifier(pl.LightningModule):
    def __init__(self, hyperparams: Config):
        super().__init__()
        self.hyperparams = hyperparams
        df = pd.read_csv('data/light_top_attention_words.csv', sep="|", index_col=False)
        self.dataset = ClusteredWordsDataset(df)
        
        self.vocab_size = self.dataset.data.shape[2]
        self.num_classes = self.dataset.labels.shape[1]
        
        # Each data element will be of shape (10, vocab_size)
        # we want to flatten it to (10 * vocab_size) so that we can
        # pass it to a fully connected layer
        self.fc1 = nn.Linear(self.vocab_size * 10, 128)
        self.fc2 = nn.Linear(128, self.num_classes)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, labels = batch
        cluster_probabilities = self(x)
        loss = nn.functional.cross_entropy(cluster_probabilities, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        cluster_probabilities = self(x)
        loss = nn.functional.cross_entropy(cluster_probabilities, labels)
        self.log("val_loss", loss)
        return loss


    def configure_optimizers(self):
        if self.hyperparams.optimizer == "adam":
            return optim.Adam(self.parameters(), lr=self.hyperparams.learning_rate)
        elif self.hyperparams.optimizer == "sgd":
            return optim.SGD(self.parameters(), lr=self.hyperparams.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer {self.hyperparams.optimizer}")

    def setup(self, stage=None) -> None:
        # Split the dataset into train and validation
        self.train_dataset, self.val_dataset = split_dataset(self.dataset, [0.8, 0.2])

        # Flatten the last two dimensions of the dataset
        # so that we can pass it to a fully connected layer
        self.train_dataset.data = self.train_dataset.data.view(
            self.train_dataset.data.shape[0], -1
        )

        self.val_dataset.data = self.val_dataset.data.view(
            self.val_dataset.data.shape[0], -1
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hyperparams.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hyperparams.batch_size, shuffle=False)


def train_experiment():
    """Trains the model and logs the results to wandb"""
    run = wandb.init()
    wandb_logger = WandbLogger()
    config = run.config
    probing_clas = ProbingClassifier(config)
    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        accelerator="gpu",
        devices=1,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
    )
    trainer.fit(
        model=probing_clas
    )

    wandb.finish()


if __name__ == "__main__":
    with open("experiments/sweep.yaml") as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config)
    wandb.agent(sweep_id, function=train_experiment)
