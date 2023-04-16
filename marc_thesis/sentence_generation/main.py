import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
import os

from nltk import sent_tokenize

import kaggle
import pickle
import re
import cleantext
from multiprocessing import Pool, cpu_count
from contextlib import closing
from loguru import logger
from marc_thesis.utils.file_manager import store_all_sentences

SENTENCE_DATASET_DIR = (
    f"{os.path.dirname(os.path.abspath(__file__))}/sentences_datasets"
)


def split_into_sentences(text: str) -> list[str]:
    # Split text into sentences
    sentences = sent_tokenize(text)
    return sentences


def extract_news_headlines() -> list[str]:
    data_dir = "{SENTENCE_DATASET_DIR}/news_headlines"
    file_name = "News_Category_Dataset_v3.json"
    kaggle_url = "rmisra/news-category-dataset"
    pickle_file = "news_headlines_sentences.pickle"

    if os.path.exists(f"{SENTENCE_DATASET_DIR}/{pickle_file}"):
        with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences

    if not os.path.exists(data_dir):
        kaggle.api.dataset_download_file_targets(
            kaggle_url, path=data_dir, unzip=True
        )

    df = pd.read_json(os.path.join(data_dir, file_name), lines=True)
    document_sentences = df["headline"].tolist()

    sentences = [
        sentence
        for paragraph in sentences
        for sentence in split_into_sentences(paragraph)
    ]

    with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def extract_new_york_times_comments() -> list[str]:
    data_dir = f"{SENTENCE_DATASET_DIR}/nyt_comments"
    kaggle_url = "aashita/nyt-comments"
    pickle_file = "nyt_comments_sentences.pickle"

    if os.path.exists(f"{SENTENCE_DATASET_DIR}/{pickle_file}"):
        with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences

    if not os.path.exists(data_dir):
        kaggle.api.dataset_download_file_targets(
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

        sentences.extend(splitted_sentences)

    with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def extract_tripadvisor_reviews() -> list[str]:
    data_dir = f"{SENTENCE_DATASET_DIR}/tripadvisor_hotel_reviews.csv"
    url = "andrewmvd/trip-advisor-hotel-reviews"
    pickle_file = "tripadvisor_hotel_reviews_sentences.pickle"
    # check if .pkl file exists
    if os.path.exists(f"{SENTENCE_DATASET_DIR}/{pickle_file}"):
        with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "rb") as f:
            sentences = pickle.load(f)
        return sentences

    # check if csv exists in sentences_datasets folder
    if not os.path.exists(data_dir):
        kaggle.api.dataset_download_file_targets(
            url, path=SENTENCE_DATASET_DIR, unzip=True
        )

    df = pd.read_csv(data_dir)
    document_sentences = df["Review"].tolist()

    sentences = [
        sentence
        for paragraph in document_sentences
        for sentence in split_into_sentences(paragraph)
    ]

    with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def mental_disorders_dataset() -> list[str]:
    data_dir = f"{SENTENCE_DATASET_DIR}/mental_disorders_reddit"
    file_name = "mental_disorders_reddit.csv"
    kaggle_url = "kamaruladha/mental-disorders-identification-reddit-nlp"
    pickle_file = "mental_disorders_reddit_sentences.pickle"

    if os.path.exists(f"{SENTENCE_DATASET_DIR}/{pickle_file}"):
        with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "rb") as f:
            corpus = pickle.load(f)
        return corpus

    if not os.path.exists(f"{data_dir}/{file_name}"):
        kaggle.api.dataset_download_file_targets(
            kaggle_url, path=data_dir, unzip=True
        )

    with open(f"{data_dir}/{file_name}", "r") as f:
        df = pd.read_csv(
            f"{data_dir}/{file_name}", dtype=str, usecols=["selftext"]
        )

    corpus = df["selftext"].tolist()

    # Make sure that sentences are actually sentences
    corpus = [
        text
        for text in corpus
        if text and type(text) == str and text != "[removed]"
    ]

    # split into sentences
    sentences = [
        sentence
        for paragraph in corpus
        for sentence in split_into_sentences(paragraph)
    ]

    with open(f"{SENTENCE_DATASET_DIR}/{pickle_file}", "wb") as f:
        pickle.dump(sentences, f)

    return sentences


def clean_string(text: str) -> str:
    # join into one string using $$$$ as separator
    clean = cleantext.clean(
        text,
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
    return clean


def clean_dataset(dataset: list[str]) -> list[str]:
    dataset = [sentence for sentence in dataset if isinstance(sentence, str)]

    logger.info("Cleaning text...")

    # Determine best chunksize
    chunksize = 2048

    with closing(Pool(processes=cpu_count())) as pool:
        sentences = [
            chunk
            for chunk in tqdm(
                pool.imap(clean_string, dataset, chunksize=chunksize),
                total=len(dataset),
                desc="Cleaning text",
            )
        ]

    return sentences


def main() -> list[str]:
    pickle_file = "training_sentences.pickle"

    kaggle.api.authenticate()

    sentences = []
    sources = []

    logger.info("Extracting sentences from mental disorders dataset")
    current_sentences = mental_disorders_dataset()
    sources += [("mental_disorders", len(current_sentences))]
    sentences += current_sentences
    logger.info(f"\tDone. {len(sentences)} sentences extracted for now")

    logger.info("Extracting sentences from Tripadvisor reviews")
    current_sentences = extract_tripadvisor_reviews()
    sources += [("tripadvisor_reviews", len(current_sentences))]
    sentences += current_sentences
    logger.info(f"\tDone. {len(sentences)} sentences extracted for now")

    logger.info("Extracting sentences from news headlines")
    current_sentences = extract_news_headlines()
    sources += [("news_headlines", len(current_sentences))]
    sentences += current_sentences
    logger.info(f"\tDone. {len(sentences)} sentences extracted for now")

    logger.info("Extracting sentences from NYT comments")
    current_sentences = extract_new_york_times_comments()
    sources += [("nyt_comments", len(current_sentences))]
    sentences += current_sentences
    logger.info(f"\tDone. {len(sentences)} sentences extracted for now")

    # logger.info("Cleaning dataset")
    # sentences = clean_dataset(sentences)
    # logger.info(f"\tDone. {len(sentences)} sentences extracted for now")

    # sentences_dict = {
    #     sentence_idx: sentence
    #     for sentence_idx, sentence in enumerate(sentences)
    # }
    # logger.info("Saving sentences to pickle file")
    # store_all_sentences(pickle_name=pickle_file, data=sentences_dict)

    # Store another pickle file with the source information
    MARC_THESIS_EXPERIMENT_FOLDER = os.environ.get(
        "MARC_THESIS_EXPERIMENT_FOLDER"
    )
    pickle.dump(
        sources,
        open(f"{MARC_THESIS_EXPERIMENT_FOLDER}/sources_info.pickle", "wb"),
    )


if __name__ == "__main__":
    main()
