import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import pickle
from tqdm import tqdm
from nltk.corpus import wordnet as wn
import torch
from loguru import logger
from file_manager import store_file_target, load_file_target

# load local model
model = AutoModelForSeq2SeqLM.from_pretrained(
    "jpwahle/t5-large-word-sense-disambiguation"
)
tokenizer = AutoTokenizer.from_pretrained(
    "jpwahle/t5-large-word-sense-disambiguation"
)


def create_query(target_word: str, senses) -> str:
    """Create a query for the T5 sense disambiguation model."""

    query = ""
    query += f'question: What is the number of the description that describes the word " {target_word} " best in the following context? '
    query += f"descriptions:\n"
    for i, sense in enumerate(senses[:-1]):
        query += f"- {sense.definition().capitalize()}\n"
    query += f"- {senses[-1].definition().capitalize()}"

    return query


def label_word_senses(
    target_word: str, senses: list[str], sentences: list[str]
) -> int:
    """Label the sense of a target_word in a given context"""

    # get 10 random sentences
    import random

    sentences = random.sample(sentences, 10)
    # Substitute target_word word by "target_word"
    query = create_query(target_word, senses)
    prompts = []
    for idx, sentence in sentences:
        prompt = query
        prompt += f"context: {sentence}"
        prompts.append(prompt)

    encoded_prompts = [
        tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts
    ]
    answers = [
        model.generate(input_ids=enc_prompt, max_new_tokens=10)
        for enc_prompt in tqdm(encoded_prompts, desc="Generating answers")
    ]
    predicted_senses = [
        tokenizer.decode(answer[0], skip_special_tokens=True)
        for answer in answers
    ]

    sense_definitons = [sense.definition() for sense in senses]

    # escape all special characters
    regexes = [re.compile(re.escape(sense[:4])) for sense in sense_definitons]
    sense_indices = []
    for predicted_sense in predicted_senses:
        for i, regex in enumerate(regexes):
            if regex.search(predicted_sense):
                sense_indices.append(i)
                break

    return sense_indices


def main():
    target_word = input("Enter the target word: ")
    sentences = load_file_target(
        target_word=target_word, file_name=f"{target_word}_sentences.pickle"
    )

    senses = wn.synsets(target_word)
    sense_frequenies = {}
    for sense in senses:
        sense_frequenies[sense] = 0
        for l in sense.lemmas():
            sense_frequenies[sense] += l.count()

    # get the senses with the top 10 frequencies
    senses = sorted(sense_frequenies, key=sense_frequenies.get, reverse=True)[
        :10
    ]

    # filter out sentences that do not contain the target_word
    regex = re.compile(rf"\b{target_word}\b")
    sentences = [
        (idx, sentence)
        for sentence, idx in zip(
            sentences["sentences"], sentences["sentence_idx"]
        )
        if regex.search(sentence) and len(tokenizer.encode(sentence)) <= 512
    ]

    sense_classification = {
        "sense": [],
        "sentence_idx": [],
    }
    # label the sense of the target_word in each sentence
    senses_indices = label_word_senses(target_word, senses, sentences)

    store_file_target(
        target_word=target_word,
        file_name=f"{target_word}_sense_classification.pickle",
        data=sense_classification,
    )


if __name__ == "__main__":
    main()
