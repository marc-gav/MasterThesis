import pandas as pd
from torch import optim, nn, utils, Tensor
import torch
from torch.utils.data import DataLoader, Dataset
import wandb
import yaml
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from wandb.sdk.wandb_config import Config

class ClusteredWordsDataset(Dataset):
    def __init__(self, data_dir: str):
        df = pd.read_csv(data_dir, sep="|", index_col=False)
        self.num_classes = len(df["cluster_label"].unique())
        self.salience_score = df["salience score"]
        words = df["word"]

        # Create a dict where every new word is mapped to an increasing index
        word_to_idx: dict = {}
        for word in words:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
        vocab_size = len(word_to_idx)
        one_hot_matrix = torch.zeros((len(words), vocab_size))
        for i, word in enumerate(words):
            one_hot_matrix[i, word_to_idx[word]] = 1

        self.data = one_hot_matrix
        self.labels = torch.tensor(
            df["cluster_label"].to_numpy(), dtype=torch.long
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def get_num_classes(self):
        """Returns the number of classes in the dataset (clusters)"""
        return self.num_classes

    def get_salience_score(self):
        """Returns the salience score of the dataset"""
        return self.salience_score
    
    def get_vocab_size(self):
        """Returns the size of the vocabulary"""
        return self.data.shape[1]


class ProbingClassifier(pl.LightningModule):
    def __init__(self, hyperparams: Config):
        super().__init__()
        self.hyperparams = hyperparams
        self.fc1 = nn.Linear(self.hyperparams.vocab_size, 100)
        self.fc2 = nn.Linear(100, self.hyperparams.num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hyperparams.lr)

PROJECT_NAME = 'Probing classifier'
DATASET = ClusteredWordsDataset("data/plant_attention_scores.csv")

def train_experiment():
    """Trains the model and logs the results to wandb"""

    run = wandb.init(project=PROJECT_NAME)
    wandb_logger = WandbLogger()
    config = run.config
    config.update({
        "vocab_size": DATASET.get_vocab_size(),
        "num_classes": DATASET.get_num_classes(),
    })
    probing_clas = ProbingClassifier(config)
    train_loader = DataLoader(DATASET, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(DATASET, batch_size=config.batch_size, shuffle=False)
    trainer = pl.Trainer(limit_train_batches=100, max_epochs=10, logger=wandb_logger)
    trainer.fit(model=probing_clas, train_dataloaders=train_loader, val_dataloaders=val_loader)

    wandb.finish()

if __name__ == "__main__":
    with open('experiments/sweep.yaml') as f:
        sweep_config = yaml.load(f, Loader=yaml.FullLoader)

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train_experiment)