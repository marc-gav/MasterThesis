from torch.utils.data import Dataset
import torch
import pandas as pd
from tqdm import tqdm
import pickle


def preprocess_labels(clusters_dict: dict) -> dict[int, int]:
    clusters_dict = {
        sentence_idx: label
        for sentence_idx, label in zip(
            clusters_dict["sentence_idx"], clusters_dict["cluster_labels"]
        )
    }

    return clusters_dict


def preprocess_data(
    salience_word_dict: dict,
) -> tuple[dict[int, torch.Tensor], dict[int, list[str]]]:
    salience_dict = {
        sentence_idx: torch.Tensor(values["salience_scores"])
        for sentence_idx, values in salience_word_dict.items()
    }

    words_dict = {
        sentence_idx: values["words"]
        for sentence_idx, values in salience_word_dict.items()
    }

    return salience_dict, words_dict


def get_vocab_dict(words_dict: dict) -> dict[str, int]:
    vocab = set()
    for sentence_idx in words_dict.keys():
        vocab.update(words_dict[sentence_idx])

    vocab = list(vocab)
    vocab_dict = {word: idx for idx, word in enumerate(vocab)}

    return vocab_dict


def prepare_for_train_val(
    data_dict: dict,
    words_dict: dict,
    labels_dict: dict,
    is_salience_weighting: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = []
    vocab_dict = get_vocab_dict(words_dict)
    data_bow = torch.zeros(len(data_dict), len(vocab_dict)).float()
    for sentence_i, sentence_idx in tqdm(
        enumerate(data_dict.keys()), desc="Preparing dataset"
    ):
        for word_j, word in enumerate(words_dict[sentence_idx]):
            if is_salience_weighting:
                word_salience = data_dict[sentence_idx][word_j]
                data_bow[sentence_i, vocab_dict[word]] += word_salience
            else:
                data_bow[sentence_i, vocab_dict[word]] = 1.0
        labels.append(labels_dict[sentence_idx])

    labels = torch.Tensor(labels).long()

    return data_bow, labels


class ClusteredBOWDataset(Dataset):
    def __init__(
        self,
        saliences: torch.Tensor,
        labels: torch.Tensor,
    ):

        self.saliences = saliences
        self.labels = labels

    def __len__(self):
        return len(self.saliences)

    def get_saliences(self):
        return self.saliences

    def get_labels(self):
        return self.labels

    def __getitem__(self, idx):
        return self.saliences[idx], self.labels[idx]
