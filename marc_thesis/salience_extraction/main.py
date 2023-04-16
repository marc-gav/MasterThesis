import torch
import transformers
from loguru import logger
from typing import Union, Tuple, List, Callable
from pprint import pprint as pp
from tqdm import tqdm
from captum.attr import (
    LayerGradientXActivation,
    LayerIntegratedGradients,
    LayerDeepLift,
    LayerGradientShap,
    LayerLRP,
)
from argparse import ArgumentParser

# import deepcopy
from copy import deepcopy

from marc_thesis.utils.file_manager import store_file_target, load_file_target


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_TOKENIZER = transformers.BertTokenizerFast.from_pretrained(
    "bert-base-uncased"
)


transformers.logging.set_verbosity_error()  # Ignore transformers warning
BERT_MASKED_MODEL: transformers.BertForMaskedLM = (
    transformers.BertForMaskedLM.from_pretrained(
        "bert-base-uncased", output_attentions=True
    )
)
transformers.logging.set_verbosity_info()  # Listen to transformers warning again

BERT_MASKED_MODEL.eval()
BERT_MASKED_MODEL.to(DEVICE)
PAD_BERT_TOKEN = (
    BERT_TOKENIZER(["[PAD]"], add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    .squeeze()
    .to(DEVICE)
)
MASK_BERT_TOKEN = (
    BERT_TOKENIZER(["[MASK]"], add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ]
    .squeeze()
    .to(DEVICE)
)
NUM_AVERAGING_LAYERS = 4

logger.info(f"Using device {DEVICE}")


def get_variable_memory_size(var):
    if type(var) == torch.Tensor:
        return var.element_size() * var.nelement()
    elif type(var) == list or type(var) == tuple:
        return sum([get_variable_memory_size(v) for v in var])
    elif type(var) == dict:
        return sum([get_variable_memory_size(v) for v in var.values()])
    else:
        logger.warning(f"Unknown type {type(var)}. Returning size 0")
        return 0

def check_device_is_available():
    if not torch.cuda.is_available():
        logger.warning("No GPU available, using CPU instead")


def objects_to_device(*objs, device):
    return [_object_to_device(obj, device) for obj in objs]


def _object_to_device(obj, device):
    if type(obj) == torch.Tensor:
        return obj.to(device)
    elif type(obj) == list:
        return [_object_to_device(v, device) for v in obj]
    elif type(obj) == tuple:
        return tuple([_object_to_device(v, device) for v in obj])
    elif (
        type(obj) == dict
        or type(obj) == transformers.tokenization_utils_base.BatchEncoding
    ):
        return {k: _object_to_device(v, device) for k, v in obj.items()}
    else:
        raise TypeError(f"Type {type(obj)} unknown")


def forward_wrapper(input_ids):
    ans = BERT_MASKED_MODEL(input_ids)
    return ans.logits


def check_everything_is_on_same_device(
    *variables: Union[torch.Tensor, list, tuple, dict]
):
    for variable in variables:
        if type(variable) == torch.Tensor:
            assert (
                variable.device.type == DEVICE.type
            ), f"Tensor is not on {DEVICE}"
        elif type(variable) == list or type(variable) == tuple:
            check_everything_is_on_same_device(*variable)
        elif type(variable) == dict:
            check_everything_is_on_same_device(*variable.values())


def batch_salience(
    salience_func,
    sentences: list[str],
    target_word: str,
    top_n: int = 10,
    batch_size=16,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_word_bert_token = (
        BERT_TOKENIZER(
            [target_word], add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        .squeeze()
        .to(DEVICE)
    )
    assert target_word_bert_token.shape == torch.Size(
        []
    ), "Target word is not a single bert subtoken"

    sentences_batches = [
        sentences[i : i + batch_size]
        for i in range(0, len(sentences), batch_size)
    ]
    salience_t = torch.zeros(len(sentences), top_n).cpu()
    salience_indices = torch.zeros(len(sentences), top_n).cpu()
    for i, sentences_batch in tqdm(
        enumerate(sentences_batches),
        total=len(sentences_batches),
        desc="Extracting salience",
    ):
        # append 10 tokens sentence to make sure that min length is 10
        sentences_batch.append(" ".join(["[PAD]"] * top_n))
        tokenized_sentences = BERT_TOKENIZER(
            sentences_batch,
            add_special_tokens=False,
            return_tensors="pt",
            padding="longest",
            truncation=False,
        )
        # remove the last sentence
        sentences_batch = sentences_batch[:-1]
        tokenized_sentences = {
            k: v[:-1] for k, v in tokenized_sentences.items()
        }

        tokenized_sentences, target_word_bert_token = objects_to_device(
            tokenized_sentences, target_word_bert_token, device=DEVICE
        )

        assert (
            tokenized_sentences["input_ids"].shape[1] <= 512
        ), "Some sentences are longer than 512 tokens"

        assert (
            tokenized_sentences["input_ids"] == target_word_bert_token
        ).sum() == len(
            sentences_batch
        ), "Some sentences do not contain the target word"

        salience_batch, salience_indices_batch = salience_func(
            tokenized_sentences,
            target_word_bert_token,
            top_n,
        )

        # To save memory, move tensors to CPU
        (
            salience_batch,
            salience_indices_batch,
            tokenized_sentences,
            target_word_bert_token,
        ) = objects_to_device(
            salience_batch,
            salience_indices_batch,
            tokenized_sentences,
            target_word_bert_token,
            device="cpu",
        )

        salience_t[i * batch_size : (i + 1) * batch_size] = salience_batch
        salience_indices[
            i * batch_size : (i + 1) * batch_size
        ] = salience_indices_batch
    return salience_t, salience_indices


def attention(
    tokenized_sentences: dict[str, torch.Tensor],
    tokenized_target: torch.Tensor,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    check_everything_is_on_same_device(tokenized_sentences, tokenized_target)
    with torch.no_grad():
        out = BERT_MASKED_MODEL(**tokenized_sentences)

    batch_attentions = out.attentions

    batch_averaged_attentions = torch.mean(
        torch.stack(batch_attentions[-NUM_AVERAGING_LAYERS:], dim=1),
        dim=[1, 2],
    )
    token_ids = tokenized_sentences["input_ids"]

    target_word_pos = torch.where(token_ids == tokenized_target)[1]

    batch_averaged_attentions_to_target = batch_averaged_attentions[
        torch.arange(batch_averaged_attentions.shape[0]), :, target_word_pos
    ]

    # Set target_word_pos or padding attention to -1
    batch_averaged_attentions_to_target[
        (token_ids == tokenized_target) + (token_ids == 0)
    ] = -1

    final_attentions, sorted_idcs = torch.topk(
        batch_averaged_attentions_to_target, k=top_n, dim=1
    )

    # Using the 2D matrix of sorted indices, get the corresponding elements of
    # the token_ids matrix
    top_token_ids = token_ids[
        torch.arange(token_ids.shape[0])[:, None], sorted_idcs
    ]

    return (deepcopy(final_attentions.cpu()), deepcopy(top_token_ids.cpu()))


def gradientXactivation(
    tokenized_sentences: dict[str, torch.Tensor],
    tokenized_target: torch.Tensor,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    check_everything_is_on_same_device(tokenized_sentences, tokenized_target)

    # Substitute the target word with the [MASK] token
    tokenized_sentences["input_ids"][
        torch.where(tokenized_sentences["input_ids"] == tokenized_target)
    ] = MASK_BERT_TOKEN

    mask_word_batch_ids, mask_word_pos = torch.where(
        tokenized_sentences["input_ids"] == MASK_BERT_TOKEN
    )
    pad_token_batch_ids, pad_token_pos = torch.where(
        tokenized_sentences["input_ids"] == PAD_BERT_TOKEN
    )

    lga = LayerGradientXActivation(
        forward_wrapper,
        BERT_MASKED_MODEL.bert.embeddings.word_embeddings,
        multiply_by_inputs=False,
    )

    # Indices needs to be a list of tuples (mask_word_pos, token_idx)
    target_indices = [
        (int(mask_word_pos[i].item()), int(tokenized_target.item()))
        for i in range(len(mask_word_pos))
    ]

    attr = lga.attribute(
        tokenized_sentences["input_ids"],
        target=target_indices,
    )

    # Detach
    attr = attr.detach()
    tokenized_sentences = {
        k: v.detach() for k, v in tokenized_sentences.items()
    }

    # Normalize last dim
    attr = torch.norm(attr, dim=-1)

    # Set mask_word_pos to -1
    attr[mask_word_batch_ids, mask_word_pos] = -1
    attr[pad_token_batch_ids, pad_token_pos] = -1

    top_n_scores, top_n_indices = torch.topk(attr, k=top_n, dim=1)

    # Get tokens corresponding to the top_n_indices
    top_n_tokens = tokenized_sentences["input_ids"][
        torch.arange(tokenized_sentences["input_ids"].shape[0])[:, None],
        top_n_indices,
    ]

    return top_n_scores, top_n_tokens


def integrated_gradient(
    tokenized_sentences: torch.Tensor,
    tokenized_target: torch.Tensor,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    check_everything_is_on_same_device(tokenized_sentences, tokenized_target)

    # Substitute the target word with the [MASK] token
    tokenized_sentences["input_ids"][
        torch.where(tokenized_sentences["input_ids"] == tokenized_target)
    ] = MASK_BERT_TOKEN

    mask_word_batch_ids, mask_word_pos = torch.where(
        tokenized_sentences["input_ids"] == MASK_BERT_TOKEN
    )
    pad_token_batch_ids, pad_token_pos = torch.where(
        tokenized_sentences["input_ids"] == PAD_BERT_TOKEN
    )

    lig = LayerIntegratedGradients(
        forward_wrapper,
        BERT_MASKED_MODEL.bert.embeddings.word_embeddings,
        multiply_by_inputs=False,
    )

    # Indices needs to be a list of tuples (mask_word_pos, token_idx)
    target_indices = [
        (int(mask_word_pos[i].item()), int(tokenized_target.item()))
        for i in range(len(mask_word_pos))
    ]

    # internal_batch_size depends on
    # the size of the input tensor
    # if num_tokens > 400, use 4
    # else use 8
    internal_batch_size = (
        4 if tokenized_sentences["input_ids"].shape[1] > 400 else 8
    )
    attr = lig.attribute(
        tokenized_sentences["input_ids"],
        target=target_indices,
        internal_batch_size=internal_batch_size,
    )

    # Detach
    attr = attr.detach()
    tokenized_sentences = {
        k: v.detach() for k, v in tokenized_sentences.items()
    }
    tokenized_target = tokenized_target.detach()

    # Normalize last dim
    attr = torch.norm(attr, dim=-1)

    # Set mask_word_pos to -1
    attr[mask_word_batch_ids, mask_word_pos] = -1
    attr[pad_token_batch_ids, pad_token_pos] = -1

    top_n_scores, top_n_indices = torch.topk(attr, k=top_n, dim=1)

    # Get tokens corresponding to the top_n_indices
    top_n_tokens = tokenized_sentences["input_ids"][
        torch.arange(tokenized_sentences["input_ids"].shape[0])[:, None],
        top_n_indices,
    ]

    return top_n_scores, top_n_tokens


def main():
    check_device_is_available()

    available_salience_funcs = {
        "attention": attention,
        "gradientXactivation": gradientXactivation,
        "integrated_gradient": integrated_gradient,
    }
    available_str = "|".join(list(available_salience_funcs.keys()))
    salience_func = input(f'Choose salience method: {available_str}:')
    target_word = input("Choose target word:")
    top_n = int(input("Choose the top N words to extract:"))
    batch_size = int(input("Choose the batch size: [Recommended: 16]"))

    assert salience_func in available_salience_funcs.keys(), f"Salience method {salience_func} not available"

    salience_func = available_salience_funcs[salience_func]

    logger.info(
        f"Extracting {salience_func} salience scores with target word {target_word} and top_n {top_n}"
    )

    sentences_dict = load_file_target(
        target_word=target_word,
        file_name=f"{target_word}_sentences.pickle",
    )
    sentence_ids = list(sentences_dict.keys())
    sentences = [sentences_dict[sentence_id] for sentence_id in sentence_ids]

    salience_t, salience_indices = batch_salience(
        salience_func,
        sentences,
        target_word,
        top_n,
        batch_size,
    )
    result_dict = {
        sentence_idx: {
            "salience_scores": salience_t[i].tolist(),
            "words": BERT_TOKENIZER.convert_ids_to_tokens(
                salience_indices[i].tolist()
            ),
            "original_sentence": sentences[i],
        }
        for i, sentence_idx in enumerate(sentence_ids)
    }

    store_file_name = (
        f"{target_word}_{salience_func}_salience_scores.pickle"
    )
    store_file_target(
        target_word=target_word,
        file_name=f"{target_word}_{salience_func}_salience_scores.pickle",
        data=result_dict,
    )

    logger.info(
        f"Done! Results stored in {store_file_name} for target word {target_word}"
    )


if __name__ == "__main__":
    main()
