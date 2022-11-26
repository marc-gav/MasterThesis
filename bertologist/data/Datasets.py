import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import os
from bertologist.utils import (
    split_into_sentences,
    bert_encode_text,
)


class NewsHeadlines:
    def __init__(self):
        self.data_dir = "datasets/News_Category_Dataset_v3.json"
        self.sentences = self.extract_sentences()

    def extract_sentences(self) -> list[str]:
        """Convert the dataset to a list of sentences."""
        df = pd.read_json(self.data_dir, lines=True)
        document_sentences = df["headline"].tolist()

        sentences = []
        # Make sure that sentences are actually sentences
        for sentence in document_sentences:
            sentences.extend(split_into_sentences(sentence))

        # Lowercase all sentences
        sentences = [sentence.lower() for sentence in sentences]
        return sentences


class NewYorkTimesComments:
    def __init__(self):
        self.data_dir = "datasets/nyt_comments"

        # If file datasets/nyt_comments_sentences.txt exists, load it
        if os.path.exists("datasets/nyt_comments_sentences.txt"):
            with open("datasets/nyt_comments_sentences.txt", "r") as f:
                self.sentences = f.read().splitlines()
        else:
            self.sentences = self.extract_sentences()

    def extract_sentences(self) -> list[str]:
        """Convert the dataset to a list of sentences."""
        sentences = []
        for file in tqdm(
            os.listdir(self.data_dir),
            desc="Extracting sentences from NYT comments",
        ):
            if not file.endswith(".csv"):
                continue

            if file.startswith("Comments"):
                df = pd.read_csv(
                    os.path.join(self.data_dir, file),
                    usecols=["commentBody"],
                    dtype=str,
                )
                document_sentences = df["commentBody"].tolist()
            elif file.startswith("Articles"):
                df = pd.read_csv(
                    os.path.join(self.data_dir, file),
                    usecols=["headline"],
                    dtype=str,
                )
                document_sentences = df["headline"].tolist()
            else:
                raise ValueError("Unexpected file in data directory")

            splitted_sentences = [
                split_into_sentences(sentence)
                for sentence in document_sentences
            ]
            sentences.extend(
                [
                    sentence.lower()
                    for sentence_list in splitted_sentences
                    for sentence in sentence_list
                ]
            )

        # Save sentences to file
        with open("datasets/nyt_comments_sentences.txt", "w") as f:
            for sentence in sentences:
                f.write(f"{sentence}\n")

        print(f"Extracted {len(sentences)} sentences from NYT comments")
        print(f"Saved sentences to datasets/nyt_comments_sentences.txt")
        return sentences


class SentencesDataset:
    def __init__(self):
        self.data_dir = "datasets/sentences.txt"
        if os.path.exists(self.data_dir):
            with open(self.data_dir, "r") as f:
                self.sentences = f.read().splitlines()
        else:
            self.sentences = self.extract_sentences()
            # store sentences in a file
            with open(self.data_dir, "w") as f:
                for sentence in self.sentences:
                    f.write(f"{sentence}\n")
            print(f"Sentences saved to {self.data_dir}")

    def extract_sentences(self) -> list[str]:
        sentences = []
        sentences += NewsHeadlines().sentences
        print(
            f"Extracted {len(NewsHeadlines().sentences)} sentences from NewsHeadlines"
        )
        sentences += TripAdvisorReviews().sentences
        print(
            f"Extracted {len(TripAdvisorReviews().sentences)} sentences from TripAdvisorReviews"
        )
        sentences += NewYorkTimesComments().sentences
        print(
            f"Extracted {len(NewYorkTimesComments().sentences)} sentences from NewYorkTimesComments"
        )
        print(f"Extracted {len(sentences)} sentences from all datasets")

        return sentences


class TripAdvisorReviews:
    def __init__(self):
        self.data_dir = "datasets/tripadvisor_hotel_reviews.csv"
        self.sentences = self.extract_sentences()

    def extract_sentences(self) -> list[str]:
        """Convert the dataset to a list of sentences."""
        df = pd.read_csv(self.data_dir)
        document_sentences = df["Review"].tolist()

        sentences = []
        # Make sure that sentences are actually sentences
        for sentence in document_sentences:
            sentences.extend(split_into_sentences(sentence))

        # Lowercase all sentences
        sentences = [sentence.lower() for sentence in sentences]
        return sentences


class ClusteredWordsDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame = None,
        data: torch.Tensor = None,
        labels: torch.Tensor = None,
    ):
        assert (df is None) != (
            data is None and labels is None
        ), "Must provide either df or data and labels"

        if data is not None and labels is not None:
            self.pre_existing_init(data, labels)
        else:
            self.df_init(df)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    def get_vocab_size(self) -> int:
        return self.data.shape[2]

    def get_num_clusters(self) -> int:
        return self.labels.shape[1]

    def df_init(self, df):
        word_to_idx: dict[str, int] = {}
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

            one_hot_matrix = torch.zeros(
                (10, vocab_size)
            )  # Fixed size 10. If the sentence is shorter, the rest is implicitly padded with 0s
            for i, word in enumerate(words):
                attention = sentence_df["salience_value"].values[i]
                one_hot_matrix[i, word_to_idx[word]] = attention
            self.data[sentence_num, :, :] = one_hot_matrix
            self.labels[sentence_num] = cluster_value

        unique_labels = torch.unique(self.labels)
        assert torch.all(
            torch.sort(unique_labels)[0]
            == torch.arange(unique_labels.max() + 1)
        ), "Labels must be consecutive integers starting at 0"
        # One-hot encode the labels
        one_hot_labels = torch.zeros((len(self.labels), 10))
        for i, label in enumerate(self.labels):
            one_hot_labels[i, label] = 1
        self.labels = one_hot_labels

    def pre_existing_init(self, data, labels):
        self.data = data
        self.labels = labels
