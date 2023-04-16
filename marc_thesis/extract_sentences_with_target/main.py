import re
from transformers import BertTokenizerFast, BertModel
import pickle
from loguru import logger
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from contextlib import closing

from marc_thesis.utils.file_manager import load_file_target, store_file_target

EXPERIMENT_FILES = os.getenv("MARC_THESIS_EXPERIMENT_FOLDER")


def main():
    target_word = input("Choose a target word: ")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    with open(
        f"{EXPERIMENT_FILES}/training_sentences.pickle", "rb"
    ) as pickle_file:
        sentences_dict: dict = pickle.load(pickle_file)
        logger.info(f"Loaded {len(sentences_dict)} sentences from file.")

    regex = re.compile(rf"\b{target_word}\b", re.IGNORECASE)
    # Tokenize the sentences
    logger.info("Tokenizing sentences...")
    target_word_id = tokenizer(target_word, add_special_tokens=False)[
        "input_ids"
    ][0]
    tokenized_sentences = {
        sentence_id: [tokenizer(sentence)["input_ids"], sentence]
        for sentence_id, sentence in tqdm(
            sentences_dict.items(), desc="Tokenizing"
        )
        if regex.search(sentence)  # Pre-filtering with whole word regex
    }

    filtered_sentences_dict = {
        sentence_id: tokenized_sentence[1]
        for sentence_id, tokenized_sentence in tqdm(
            tokenized_sentences.items(), desc="Filtering"
        )
        # if target_word_id appears only once in the sentence and sentence is <= 512 tokens
        if (
            tokenized_sentence[0].count(target_word_id) == 1
            and len(tokenized_sentence[0]) <= 512
        )
    }

    logger.info(
        f"Found {len(filtered_sentences_dict)} sentences with the target word"
    )
    logger.info("Notes: One single occurrence of the target word is required")
    logger.info("       Sentences with more than 512 BERT tokens are skipped")

    store_file_target(
        target_word=target_word,
        file_name=f"{target_word}_sentences.pickle",
        data=filtered_sentences_dict,
    )


if __name__ == "__main__":
    main()
