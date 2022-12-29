from torch import optim, nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import pytorch_lightning as pl
from wandb.sdk.wandb_config import Config
import pandas as pd
import wandb


def log_extra_info(cluster_probabilities, labels, epoch, step, train_val):
    probabilities = list(cluster_probabilities[0].detach().cpu().numpy())
    labels = list(labels[0].detach().cpu().numpy())
    cluster_nums = list(range(len(labels)))
    data = [
        [data_point[0], data_point[1], data_point[2]]
        for data_point in zip(probabilities, labels, cluster_nums)
    ]
    table = wandb.Table(
        data=data, columns=["probability", "truth", "cluster_num"]
    )
    wandb.log(
        {
            f"cluster_probabilities_barchart_{train_val}": wandb.plot.bar(
                table, "cluster_num", "probability"
            )
        }
    )

    # TODO: Log the original sentence as well


class BaseProbingClassifier(pl.LightningModule):
    def __init__(self, hyperparams: Config, input_size, num_clusters):
        super().__init__()
        self.hyperparams = hyperparams

        self.input_size = input_size
        self.num_clusters = num_clusters

    def forward(self, x):
        raise NotImplementedError("Implement this in the child class")

    def training_step(self, batch, batch_idx):
        x, labels = batch
        cluster_probabilities = self(x)

        loss = nn.functional.cross_entropy(cluster_probabilities, labels)

        # every 5 epochs, log the cluster probabilities
        current_epoch = self.current_epoch
        if current_epoch % 5 == 0 and batch_idx == 0:
            log_extra_info(
                cluster_probabilities,
                labels,
                self.current_epoch,
                batch_idx,
                "train",
            )

        # log accuracy
        preds = cluster_probabilities.argmax(dim=1)
        acc = (preds == labels.argmax(dim=1)).float().mean()
        self.log("train_acc", acc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        cluster_probabilities = self(x)

        # de-biass
        cluster_probabilities = cluster_probabilities * self.class_weights

        loss = nn.functional.cross_entropy(cluster_probabilities, labels)

        current_epoch = self.current_epoch
        if current_epoch % 5 == 0 and batch_idx == 0:
            log_extra_info(
                cluster_probabilities,
                labels,
                self.current_epoch,
                batch_idx,
                "val",
            )

        # log accuracy
        preds = cluster_probabilities.argmax(dim=1)
        acc = (preds == labels.argmax(dim=1)).float().mean()
        self.log("val_acc", acc)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        if self.hyperparams.optimizer == "adam":
            return optim.Adam(
                self.parameters(), lr=self.hyperparams.learning_rate
            )
        elif self.hyperparams.optimizer == "sgd":
            return optim.SGD(
                self.parameters(), lr=self.hyperparams.learning_rate
            )
        else:
            raise ValueError(f"Unknown optimizer {self.hyperparams.optimizer}")


class ProbingClassifier(BaseProbingClassifier):
    def __init__(
        self, hyperparams: Config, input_size, num_clusters, class_weights
    ):
        super().__init__(
            hyperparams,
            num_clusters=num_clusters,
            input_size=input_size,
        )
        self.class_weights = class_weights

        # 3 layers with regularization
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_clusters)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # debias
        x = x * self.class_weights
        return x
