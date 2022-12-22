import re
from collections import defaultdict
import torch
import nltk

import pandas as pd
from bertologist.utils import (
    extract_vector_hidden_state,
    target_word_is_in_sentence,
    bert_encode_text,
)
from tqdm import tqdm


def get_embeddings_from_target_word_in_sentences(
    corpus: list[str], target_word: str
):
    """Extract the word embedding of the target word in each sentence. Return a
    list of tuples of the form (sentence, embedding). Can also store it in a csv
    file.

    Args:
        target_word (str): Target word
        corpus (list[str]): Corpus

    Returns:
        list[tuple[str, torch.Tensor]]: list of word embeddings
    and their corresponding sentence
    """

    _, tokenized_target_word = bert_encode_text(
        target_word, special_tokens=False
    )
    sentences = []
    target_word_embeddings = []
    sentences_with_target_word = [
        target_word_is_in_sentence(sentence, target_word)
        for sentence in tqdm(
            corpus, desc="Filtering sentences with target word"
        )
    ]

    corpus_with_target_word = [
        sentence
        for sentence, target_word_present in zip(
            corpus, sentences_with_target_word
        )
        if target_word_present
    ]

    print("Encoding sentences with BERT")
    bert_encoded_corpus = [
        (pos, bert_encode_text(sentence, special_tokens=True)[1])
        for pos, sentence in enumerate(corpus_with_target_word)
    ]
    print(f"Sentences with BERT have been encoded")

    print("Filtering long sentences")
    bert_encoded_corpus = [
        (pos, bert_encoded_text_item)
        for pos, bert_encoded_text_item in bert_encoded_corpus
        if len(bert_encoded_text_item) <= 512
    ]
    print(f"Long sentences have been filtered")

    for pos, tokenized_sentence in tqdm(
        bert_encoded_corpus, desc="Extracting embeddings"
    ):
        target_word_vectors = extract_vector_hidden_state(
            tokenized_sentence, tokenized_target_word
        )
        for target_word_vector in target_word_vectors:
            target_word_embeddings.append(target_word_vector)
            sentences.append(corpus_with_target_word[pos])

    # stack the embeddings
    target_word_embeddings = torch.stack(target_word_embeddings).squeeze()
    return target_word_embeddings, sentences
