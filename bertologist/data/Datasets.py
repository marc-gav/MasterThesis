import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import os

from bertologist.utils import (
    split_into_sentences,
)
import kaggle
import pickle
import re
import cleantext
from multiprocessing import Pool, cpu_count
from contextlib import closing


def extract_news_headlines() -> list[str]:
    data_dir = "datasets/news_headlines"
    file_name = "News_Category_Dataset_v3.json"
    kaggle_url = "rmisra/news-category-dataset"
    pickle_file = "news_headlines_sentences.pickle"

    if os.path.exists(f"datasets/{pickle_file}"):
        with open(f"datasets/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences

    if not os.path.exists(data_dir):
        kaggle.api.dataset_download_files(
            kaggle_url, path=data_dir, unzip=True
        )

    df = pd.read_json(os.path.join(data_dir, file_name), lines=True)
    document_sentences = df["headline"].tolist()

    sentences = []
    # Make sure that sentences are actually sentences
    for sentence in tqdm(document_sentences, desc="Extracting sentences"):
        sentences.extend(split_into_sentences(sentence))

    # Lowercase all sentences
    sentences = [sentence.lower() for sentence in sentences]

    with open(f"datasets/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def extract_new_york_times_comments() -> list[str]:
    data_dir = "datasets/nyt_comments"
    kaggle_url = "aashita/nyt-comments"
    pickle_file = "nyt_comments_sentences.pickle"

    if os.path.exists(f"datasets/{pickle_file}"):
        with open(f"datasets/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences

    if not os.path.exists(data_dir):
        kaggle.api.dataset_download_files(
            kaggle_url, path=data_dir, unzip=True
        )

    sentences = []
    for file in tqdm(
        os.listdir(data_dir),
        desc="Extracting sentences from NYT comments",
    ):
        if not file.endswith(".csv"):
            continue

        if file.startswith("Comments"):
            df = pd.read_csv(
                os.path.join(data_dir, file),
                usecols=["commentBody"],
                dtype=str,
            )
            document_sentences = df["commentBody"].tolist()
        elif file.startswith("Articles"):
            df = pd.read_csv(
                os.path.join(data_dir, file),
                usecols=["headline"],
                dtype=str,
            )
            document_sentences = df["headline"].tolist()
        else:
            raise ValueError(f"Unexpected file '{file}' in data directory")

        splitted_sentences = [
            split_into_sentences(sentence) for sentence in document_sentences
        ]
        sentences.extend(
            [
                sentence.lower()
                for sentence_list in splitted_sentences
                for sentence in sentence_list
            ]
        )

    with open(f"datasets/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def extract_tripadvisor_reviews() -> list[str]:
    data_dir = "datasets/tripadvisor_hotel_reviews.csv"
    url = "andrewmvd/trip-advisor-hotel-reviews"
    pickle_file = "tripadvisor_hotel_reviews_sentences.pickle"
    # check if .pkl file exists
    if os.path.exists(f"datasets/{pickle_file}"):
        with open(f"datasets/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences

    # check if csv exists in datasets folder
    if not os.path.exists(data_dir):
        kaggle.api.dataset_download_files(url, path="datasets", unzip=True)

    df = pd.read_csv(data_dir)
    document_sentences = df["Review"].tolist()

    sentences = []
    # Make sure that sentences are actually sentences
    for sentence in document_sentences:
        sentences.extend(split_into_sentences(sentence))

    # Lowercase all sentences
    sentences = [sentence.lower() for sentence in sentences]

    with open(f"datasets/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def mental_disorders_dataset() -> list[str]:
    data_dir = "datasets/mental_disorders_reddit"
    file_name = "mental_disorders_reddit.csv"
    kaggle_url = "kamaruladha/mental-disorders-identification-reddit-nlp"
    pickle_file = "mental_disorders_reddit_sentences.pickle"

    if os.path.exists(f"datasets/{pickle_file}"):
        with open(f"datasets/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences

    if not os.path.exists(f"{data_dir}/{file_name}"):
        kaggle.api.dataset_download_files(
            kaggle_url, path=data_dir, unzip=True
        )

    with open(f"{data_dir}/{file_name}", "r") as f:
        df = pd.read_csv(
            f"{data_dir}/{file_name}", dtype=str, usecols=["selftext"]
        )

    sentences = df["selftext"].tolist()

    with open(f"datasets/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def clean_string(text: list[str]) -> list[str]:
    clean_sentences = []
    for sentence in tqdm(text):
        clean = cleantext.clean(
            sentence,
            fix_unicode=False,  # fix various unicode errors
            to_ascii=False,  # transliterate to closest ASCII representation
            lower=True,  # lowercase text
            no_line_breaks=True,  # fully strip line breaks as opposed to only normalizing them
            no_urls=True,  # replace all URLs with a special token
            no_emails=False,  # replace all email addresses with a special token
            no_phone_numbers=False,  # replace all phone numbers with a special token
            no_numbers=False,  # replace all numbers with a special token
            no_digits=False,  # replace all digits with a special token
            no_currency_symbols=False,  # replace all currency symbols with a special token
        )
        clean_sentences.extend(split_into_sentences(clean))
    return clean_sentences


def clean_dataset(dataset: list[str]) -> list[str]:
    # search for elements of the list that are not strings
    dataset = [sentence for sentence in dataset if isinstance(sentence, str)]

    num_cpu = cpu_count()
    # split dataset in 4 chunks
    stride = len(dataset) // num_cpu
    dataset_chunks = [
        dataset[i : i + stride] for i in range(0, len(dataset), stride)
    ]

    print(f"There is a total of {len(dataset_chunks)} chunks.")
    print("They have a size of:")
    for i, chunk in enumerate(dataset_chunks):
        print(f"Chunk {i + 1}: {len(chunk)} sentences")

    # add leftover sentences to last chunk
    dataset_chunks[-1].extend(dataset[stride * num_cpu :])

    print("Cleaning text...")

    sentences = []
    with closing(Pool(processes=num_cpu)) as pool:
        for i, chunk in enumerate(pool.imap(clean_string, dataset_chunks)):
            sentences.extend(chunk)
            print(f"Chunk {i + 1} of {len(dataset_chunks)} cleaned")

    return sentences


def load_training_sentences() -> list[str]:
    pickle_file = "training_sentences.pickle"

    kaggle.api.authenticate()

    if os.path.exists(f"datasets/{pickle_file}"):
        with open(f"datasets/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences
    else:
        sentences = []

        print("Extracting sentences from mental disorders dataset")
        sentences += mental_disorders_dataset()
        print(f"Done. {len(sentences)} sentences extracted for now")

        print("Extracting sentences from Tripadvisor reviews")
        sentences += extract_tripadvisor_reviews()
        print(f"Done. {len(sentences)} sentences extracted for now")

        print("Extracting sentences from news headlines")
        sentences += extract_news_headlines()
        print(f"Done. {len(sentences)} sentences extracted for now")

        print("Extracting sentences from NYT comments")
        sentences += extract_new_york_times_comments()
        print(f"Done. {len(sentences)} sentences extracted for now")

        print("Cleaning dataset")
        sentences = clean_dataset(sentences)
        print(f"Done. {len(sentences)} sentences extracted for now")

        print("Saving dataset...")
        # picke sentences
        with open(f"datasets/{pickle_file}", "wb") as f:
            pickle.dump(sentences, f)
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
        for sentence_num, sentence_idx in tqdm(
            enumerate(self.sentences),
            total=len(self.sentences),
            desc="Loading data and labels into torch arrays",
        ):
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
