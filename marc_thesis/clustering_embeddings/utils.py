import torch
import transformers

bert_tokenizer = transformers.BertTokenizer.from_pretrained(
    "bert-base-uncased"
)
bert_model = transformers.BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True,
)


def bert_encode_text(
    text: str,
    special_tokens: bool,
) -> tuple[list[str], torch.Tensor]:
    """Encode the text using BERT tokenizer

    Args:
        text (str): Text to encode
        tokenizer (transformers.PreTrainedTokenizerFast): BERT tokenizer
        special_tokens (bool): Whether to add BERT special tokens or not

    Returns tuple[list[str], torch.Tensor]: A tuple of:
        - list[str]: Tokenized text
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


def extract_vector_hidden_state(
    sentence_token_ids: torch.Tensor, target_word_token_ids: torch.Tensor
) -> list[torch.Tensor]:
    """Extract the target word vector hidden state from the last layer of the BERT model.

        Note: In case multiple instances of the target word are found, the list will contain
    multiple vectors
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
                        [list(range(i, i + len(target_word_token_ids)))]
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
