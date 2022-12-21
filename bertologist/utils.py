import csv
import re
from collections import defaultdict
from pprint import pprint as pp
from typing import Generator, Iterator, Union, List, Tuple

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers

transformers.logging.set_verbosity_error()
#nltk.download("punkt")
NUM_AVERAGING_LAYERS = 4

bert_tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-base-uncased"
)
bert_model = transformers.BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True,
)


def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences.

    Args:
        text (str): Text to split

    Returns:
        List[str]: List of sentences
    """
    return nltk.sent_tokenize(text)


def target_word_is_in_sentence(sentence: str, target_word: str) -> bool:
    """Check if the target word is in the sentence.

    Args:
        sentence (str): Sentence
        target_word (str): Target word

    Returns:
        bool: True if the target word is in the sentence
    """
    return nltk.word_tokenize(sentence).count(target_word) > 0


def bert_encode_text(
    text: str,
    special_tokens: bool,
) -> Tuple[List[str], torch.Tensor]:
    """Encode the text using BERT tokenizer

    Args:
        text (str): Text to encode
        tokenizer (transformers.PreTrainedTokenizerFast): BERT tokenizer
        special_tokens (bool): Whether to add BERT special tokens or not

    Returns Tuple[List[str], torch.Tensor]: A Tuple of:
        - List[str]: Tokenized text
        - torch.Tensor: Token representation as BERT ids
    """
    batch_encoding_tokenized: transformers.BatchEncoding = bert_tokenizer(
        text, return_tensors="pt", add_special_tokens=special_tokens
    )
    bert_input_ids: torch.Tensor = batch_encoding_tokenized[
        "input_ids"
    ].squeeze()
    tokenized_text = bert_tokenizer.convert_ids_to_tokens(
        bert_input_ids.tolist()
    )
    return tokenized_text, bert_input_ids


def _extract_attention_averaging_layers_and_heads(
    attention_layers: Tuple[torch.Tensor], last_n_layers: int
) -> torch.Tensor:
    """Extract the attention from the last n layers. Averaging over all the heads.

    Args:
        attention_layers (Tuple[torch.Tensor]): Attention from all the layers
        last_n_layers (int): Number of last layers to average over

    Returns:
        torch.Tensor: Averaged attention from the last n layers
    """
    assert last_n_layers <= len(
        attention_layers
    ), "last_n_layers should be <= len(attention_layers)"
    attention_of_last_n_layers = attention_layers[-last_n_layers:]

    last_n_layers_tensor = torch.stack(
        attention_of_last_n_layers, dim=0
    ).squeeze()

    if NUM_AVERAGING_LAYERS > 1:
        return torch.mean(last_n_layers_tensor, dim=[0, 1])
    else:
        return torch.mean(last_n_layers_tensor, dim=[0])


def _average_attention_for_multiple_token_words(
    attention: torch.Tensor,
    tokenized_text: List[str],
) -> torch.Tensor:
    """Averages the attention for multiple token words. This is done by
    averaging the rows and columns that are spanned by the multiple
    token words into a single row and column.


    Args:
        attention (torch.Tensor): Attention matrix
        tokenized_text (List[str]): Tokenized text

    Returns:
        torch.Tensor: Averaged attention matrix
    """

    attention_df = pd.DataFrame(attention.numpy())
    mask = torch.tensor(
        [1 for word in tokenized_text if not re.match(r"##", word)],
        dtype=torch.bool,
    )
    maks_series = pd.Series(mask.tolist())
    attention_df = attention_df.groupby(((maks_series).cumsum())).mean()
    attention_df = attention_df.T.groupby(((maks_series).cumsum())).mean()

    return torch.tensor(attention_df.to_numpy())


def _create_target_word_mask(
    tokenized_text: List[str], target_word: str
) -> Tuple[torch.Tensor, List[str]]:
    """Create a mask for the target word where everything is false except
    the target word.

    Assumptions:
        - Multiple token words have been averaged by this point.
        - One single sentence is passed in. BERT can deal with two sentences
        at a time but this function does not.

    Args:
        sentence (str): Sentence
        target_word (str): Target word

    Returns:
        Tuple[torch.Tensor, List[str]]: A Tuple of:
            - torch.Tensor: Mask for the target word
            - List[str]: Tokenized text
    """

    sentence_list = []
    current_word = tokenized_text[0]
    for word in tokenized_text[1:]:
        if word.startswith("##"):
            current_word += word[2:]
        else:
            sentence_list.append(current_word)
            current_word = word

    mask = torch.tensor(
        [True if word == target_word else False for word in sentence_list]
    )

    return mask, sentence_list


def _extract_attention_to_target(
    avg_attention: torch.Tensor,
    target_word_mask: torch.Tensor,
    sentence_words_list: List[str],
) -> Tuple[torch.Tensor, List[str]]:
    """Extract the attention of other words to the target word. If multiple instances of the
    target word are present, expect multiple columns in the attention to target word tensor.

    Assumptions:
        - Attention matrix has averaged the multiple
        word tokens into one single row.
        - Attention matrix has averaged the multiple
        word tokens into one single column.

    Args:
        avg_attention (torch.Tensor): Averaged attention matrix
        target_word_mask (torch.Tensor): Mask for the target word
        sentence_words_list (List[str]): Words of the sentence and special tokens

    Returns:
        Tuple[torch.Tensor, List[str]]: A Tuple of:
            - torch.Tensor: Attention to the target word
            - List[str]: Sentence without the target words
    """

    target_indices = torch.nonzero(target_word_mask).squeeze()

    non_target_indices = torch.nonzero(~target_word_mask).squeeze()

    attention_to_target = avg_attention[non_target_indices, :][
        :, target_indices
    ]

    # Extract non_target_indices from sentence_words_list
    non_target_words = [sentence_words_list[i] for i in non_target_indices]

    # Make sure that attention_to_target always has two dimensions
    if len(attention_to_target.shape) == 1:
        attention_to_target = attention_to_target.unsqueeze(0)
    else:
        attention_to_target = attention_to_target.T

    return attention_to_target, non_target_words


def _create_multiple_word_token_mask(
    tokenized_text: List[str],
) -> torch.Tensor:
    """Create a mask for the tokens that are part of a multiple word token.

    Args:
        tokenized_text (List[str]): Tokenized text

    Returns:
        torch.Tensor: Mask for the tokens that are part of a multiple word token
    """

    multiple_word_token_mask1 = torch.tensor(
        [1 if token.startswith("##") else 0 for token in tokenized_text],
        dtype=torch.bool,
    )

    multiple_word_token_mask2 = torch.tensor(
        [
            1 if multiple_word_token_mask1[i + 1] else 0
            for i in range(len(multiple_word_token_mask1) - 1)
        ],
        dtype=torch.bool,
    )
    # Add the last token
    multiple_word_token_mask2 = torch.cat(
        [multiple_word_token_mask2, torch.tensor([False])]
    )

    # OR
    multiple_word_token_mask = torch.logical_or(
        multiple_word_token_mask1, multiple_word_token_mask2
    )

    return multiple_word_token_mask


def get_attention_to_target_word_in_sentence(
    sentence: str, target_word: str
) -> Tuple[torch.Tensor, List[str]]:
    """Get the attention of other words to the target word in a sentence.

    Args:
        sentence (str): Sentence
        target_word (str): Target word

    Returns:
        Tuple[torch.Tensor, List[str]]: A Tuple of:
            - torch.Tensor: Attention to the target word
            - List[str]: Sentence without the target words
    """

    tokenized_text, text_bert_ids = bert_encode_text(
        sentence, special_tokens=True
    )

    with torch.no_grad():
        out = bert_model(text_bert_ids.unsqueeze(0))
        attentions = out.attentions

    avg_attention_layers = _extract_attention_averaging_layers_and_heads(
        attentions, last_n_layers=NUM_AVERAGING_LAYERS
    )

    avg_attention_layers_avg_multiple_token_words = (
        _average_attention_for_multiple_token_words(
            avg_attention_layers,
            tokenized_text,
        )
    )

    target_word_mask, sentence_list = _create_target_word_mask(
        tokenized_text, target_word
    )

    avg_attention_to_target, non_target_words = _extract_attention_to_target(
        avg_attention_layers_avg_multiple_token_words,
        target_word_mask,
        sentence_list,
    )

    return avg_attention_to_target, non_target_words


def get_top_n_attention_words_to_target(
    sentence: str, target_word: str, n: int = -1
) -> Iterator[List]:
    """Get the top n words that have the highest attention to the target word.

    Args:
        sentence (str): Sentence
        target_word (str): Target word
        n (int, optional): Number of words to return. Defaults to 5.

    Yields for each occurence of the target word in the sentence:
        pd.DataFrame: Dataframe with the top n words,
            one List element per occurrence of the target word in the sentence.
    """

    (
        attention_matrix_per_word_ocurrence,
        non_target_words,
    ) = get_attention_to_target_word_in_sentence(sentence, target_word)

    for attention_matrix in attention_matrix_per_word_ocurrence:

        word_and_attention = List(
            zip(non_target_words, attention_matrix.squeeze().tolist())
        )
        word_and_attention = sorted(
            word_and_attention, key=lambda x: x[1], reverse=True
        )

        yield word_and_attention[:n]


def extract_vector_hidden_state(
    sentence_token_ids: torch.Tensor, target_word_token_ids: torch.Tensor
) -> List[torch.Tensor]:
    """Extract the target word vector hidden state from the last layer of the BERT model.
    In case multiple instances of the target word are found, the List will contain
    multiple vectors.

    Returns:
        List[torch.Tensor]: List of vector hidden states, it is a List
        because more than one target word can appear in a sentence.
    """

    list_target_word_indices = []

    if len(target_word_token_ids.shape) == 0:  # Single token
        for i, token_id in enumerate(sentence_token_ids):
            if token_id == target_word_token_ids:
                list_target_word_indices.append(torch.tensor(i))
    else:  # Multiple tokens
        # Rolling window
        for i in range(
            len(sentence_token_ids) - len(target_word_token_ids) + 1
        ):
            if torch.equal(
                sentence_token_ids[i : i + len(target_word_token_ids)],
                target_word_token_ids,
            ):
                list_target_word_indices.append(
                    torch.tensor(
                        [List(range(i, i + len(target_word_token_ids)))]
                    )
                )

    # Get the hidden state of the last layer of the BERT model
    with torch.no_grad():
        out = bert_model(sentence_token_ids.unsqueeze(0))
        last_hidden_state = out.last_hidden_state.squeeze()

    list_target_word_hidden_states = []
    for target_word_indices in list_target_word_indices:
        target_word_hidden_states = last_hidden_state[target_word_indices, :]
        if len(target_word_hidden_states.shape) == 1:
            target_word_hidden_states = target_word_hidden_states.unsqueeze(0)
        else:  # Average them
            target_word_hidden_states = torch.mean(
                target_word_hidden_states, dim=0
            )
        list_target_word_hidden_states.append(target_word_hidden_states)

    return list_target_word_hidden_states
