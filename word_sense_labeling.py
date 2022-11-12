import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# load local model
model = AutoModelForSeq2SeqLM.from_pretrained(
    "huggingface_models/t5-word-sense-disambiguation", local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "huggingface_models/t5-word-sense-disambiguation", local_files_only=True
)


def create_query(target_word: str, senses: list[str], context: str) -> str:
    """Create a query for the T5 sense disambiguation model.

    Args:
        target_word (str): The target_word to disambiguate.
        senses (list[str]): A list of senses for the target_word.
        context (str): The context in which the target_word appears.

    Returns:
        str: A query for the T5 sense disambiguation model.
    """

    query = ""
    query += f'question: Which description describes the word " {target_word} " best in the following context? '
    query += f"descriptions: [ "
    for i, sense in enumerate(senses[:-1]):
        query += f'" {sense.capitalize()} " , '
    query += f'" {senses[-1].capitalize()} " ] '
    context = re.sub(rf"\b{target_word}\b", f'" {target_word} "', context)
    query += f"context: {context}"

    return query


def label_word_sense(target_word: str, senses: list[str], context: str) -> int:
    """Label the sense of a target_word in a given context.

    Args:
        target_word (str): The target_word to disambiguate.
        senses (list[str]): A list of senses for the target_word.
        context (str): The context in which the target_word appears.

    Returns:
        int: The index of the sense that best describes the target_word in the given context.
    """

    # Substitute target_word word by "target_word"
    query = create_query(target_word, senses, context)
    input_ids = tokenizer.encode(query, return_tensors="pt")
    answer = model.generate(
        input_ids=input_ids,
        max_length=135,  # TODO: figure out how to calculate this
    )
    predicted_sense = tokenizer.decode(answer[0])
    special_tokens = tokenizer.all_special_tokens
    predicted_sense = re.sub(
        f"({'|'.join(special_tokens)})", "", predicted_sense
    )

    # Obtain the index of the senses with the most amount of common
    # words with the predicted sense
    common_words = [
        len(set(predicted_sense.split()).intersection(set(sense.split())))
        for sense in senses
    ]
    sense_index = common_words.index(max(common_words))
    return sense_index
