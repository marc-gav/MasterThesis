import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from contextlib import closing
import os
import pickle
import re
from loguru import logger

from marc_thesis.clustering_embeddings.utils import (
    extract_vector_hidden_state,
    bert_encode_text,
)
from marc_thesis.utils.file_manager import store_file_target, load_file_target


def _extract_embeddings(bert_encoded_corpus, tokenized_target_word):
    embeddings_dict = {
        "target_word_embeddings": [],
        "sentence_idx": [],
    }

    for sentence_idx, tokenized_sentence in tqdm(
        bert_encoded_corpus, desc="Extracting embeddings"
    ):
        target_word_vectors = extract_vector_hidden_state(
            tokenized_sentence, tokenized_target_word
        )
        for target_word_vector in target_word_vectors:
            embeddings_dict["target_word_embeddings"].append(
                target_word_vector
            )
            embeddings_dict["sentence_idx"].append(sentence_idx)

    return embeddings_dict


def extract_bert_embeddings_of_word(
    sentences_dict: dict, target_word: str
) -> dict:
    """Extract the word embedding of the target word in each sentence.

    Assumption:
        - The target word is in the sentences
    """

    sentences = sentences_dict["sentences"]
    indices = sentences_dict["sentence_idx"]

    _, tokenized_target_word = bert_encode_text(
        target_word, special_tokens=False
    )

    bert_encoded_corpus = [
        (idx, bert_encode_text(sentence, special_tokens=True)[1])
        for sentence, idx in tqdm(
            zip(sentences, indices),
            desc="Encoding sentences with BERT",
            total=len(sentences),
        )
    ]

    bert_encoded_corpus = [
        (idx, bert_encoded_text_item)
        for idx, bert_encoded_text_item in tqdm(
            bert_encoded_corpus,
            desc="Filtering long sentences",
            total=len(bert_encoded_corpus),
        )
        if len(bert_encoded_text_item) <= 512
    ]

    embeddings_dict = _extract_embeddings(
        bert_encoded_corpus, tokenized_target_word
    )

    store_file_target(
        target_word, f"{target_word}_embeddings.pickle", embeddings_dict
    )

    return embeddings_dict
