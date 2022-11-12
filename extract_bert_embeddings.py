import re
from collections import defaultdict

import nltk
import pandas as pd
from bert_utils import (
    bert_model,
    bert_tokenizer,
    extract_vector_hidden_state,
    split_into_sentences,
    target_word_is_in_sentence,
)
from tqdm import tqdm


def get_embeddings_from_target_word_in_sentences(
    target_word: str, corpus: list[str], store_csv: bool = True
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
    target_word = "plant"

    # stores tuples of (sentence, word embedding vector)
    sentence_and_target_word_vector = []

    for headline in tqdm(corpus):
        sentences = split_into_sentences(headline)
        for sentence in sentences:
            if not target_word_is_in_sentence(sentence, target_word):
                continue

            # Extract hidden states from the last layer of the target word
            # and the non-target words
            target_word_vectors = extract_vector_hidden_state(
                sentence, target_word
            )
            for target_word_vector in target_word_vectors:
                sentence_and_target_word_vector.append(
                    (sentence, target_word_vector)
                )

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
    news_dataset = pd.read_json("data/plant_news.json", lines=True)
    news_descriptions = news_dataset.short_description.tolist()

    get_embeddings_from_target_word_in_sentences("plant", news_descriptions)
