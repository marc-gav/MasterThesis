import re
from collections import defaultdict

import nltk
import pandas as pd
from bert_utils.bert_utils import (
    bert_model,
    bert_tokenizer,
    extract_vector_hidden_state,
    split_into_sentences,
    target_word_is_in_sentence,
    bert_encode_text,
)
from tqdm import tqdm


def get_embeddings_from_target_word_in_sentences(
    corpus: list[str], target_word: str, store_csv: bool = False
):
    """Extract the word embedding of the target word in each sentence. Return a
    list of tuples of the form (sentence, embedding). Can also store it in a csv
    file.

    Args:
        target_word (str): Target word
        corpus (list[str]): Corpus

    Returns:
        list[Tuple[str, torch.Tensor]]: List of word embeddings
    and their corresponding sentence
    """

    # stores tuples of (sentence, word embedding vector)
    sentence_and_target_word_vector = []

    for sentence in tqdm(corpus):
        if not target_word_is_in_sentence(sentence, target_word):
            continue

        _, bert_token_ids = bert_encode_text(sentence, special_tokens=True)
        if len(bert_token_ids) > 512:
            continue

        # Extract hidden states from the last layer of the target word
        # and the non-target words
        target_word_vectors = extract_vector_hidden_state(sentence, target_word)
        for target_word_vector in target_word_vectors:
            sentence_and_target_word_vector.append((sentence, target_word_vector))

    if store_csv:
        # Convert target_word_vector to string
        sentence_and_target_word_vector = [
            (sentence, target_word_vector.tolist())
            for sentence, target_word_vector in sentence_and_target_word_vector
        ]
        df = pd.DataFrame(
            sentence_and_target_word_vector,
            columns=["sentence", "target_word_vector"],
        )

        df.to_csv(
            f"data/bert_embeddings_of_word_{target_word}_in_sentence.csv",
            sep="|",
        )

        print(
            f'Stored csv file "data/bert_embeddings_of_word_{target_word}_in_sentence.csv"'
        )

    return sentence_and_target_word_vector


if __name__ == "__main__":
    news_dataset = pd.read_json("data/News_Category_Dataset_v3.json", lines=True)
    news_descriptions = news_dataset.short_description.tolist()

    get_embeddings_from_target_word_in_sentences("one", news_descriptions)
