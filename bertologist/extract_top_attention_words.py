from bert_utils.bert_utils import get_top_n_attention_words_to_target
from tqdm import tqdm
import os
import re
import pandas as pd


def extract_context_attention(
    target_word: str,
    clustered_embeddings_df: pd.DataFrame,
    store_csv: bool = True,
):
    """Extract the attention weights of the context words towards the target word.

    Args:
        target_word (str): Target word
        clustered_embeddings_df (pd.DataFrame): Dataframe of clustered embeddings
        store_csv (bool, optional): Store the attention weights in a csv file. Defaults to True.

    Returns:
        # TODO: define return type
    """

    # Remove repeated sentences
    clustered_embeddings_df.drop_duplicates(subset=["sentence"], inplace=True)

    df_top_attention_all_sentences = pd.DataFrame(
        columns=[
            "sentence",
            "word",
            "attention",
            "cluster_label",
            "sentence_index",
        ]
    )

    sentence_number = 0  # I know it would make more sense
    # to use the index of the row, but in case a target word
    # appears multiple times in a sentence, I treat them as separate sentences
    for _, row in tqdm(
        clustered_embeddings_df.iterrows(),
        desc=f"Extracting top attention words for {target_word}",
        total=len(clustered_embeddings_df),
    ):
        sentence = row["sentence"]
        for top_n_attention in get_top_n_attention_words_to_target(
            sentence, target_word, n=10  # Top 10 words
        ):

            top_n_attention["sentence_index"] = sentence_number
            top_n_attention["salience_score"] = "attention"
            top_n_attention["sentence"] = sentence
            top_n_attention["cluster_label"] = row["cluster_label"]

            df_top_attention_all_sentences = pd.concat(
                [
                    df_top_attention_all_sentences,
                    top_n_attention,
                ]
            )

            sentence_number += 1

    if store_csv:
        df_top_attention_all_sentences.to_csv(
            f"data/{target_word}_top_attention_words.csv",
            sep="|",
            index=False,
        )
        print(
            f'Top attention words saved to "data/{target_word}_top_attention_words.csv"'
        )

    return df_top_attention_all_sentences


if __name__ == "__main__":
    for file in os.listdir("data"):
        if re.match(r"(.*)_embeddings_clustered.csv", file):
            target_word = re.match(
                r"(.*)_embeddings_clustered.csv", file
            ).group(1)
            clustered_embeddings_df = pd.read_csv(
                f"data/{target_word}_embeddings_clustered.csv",
                sep="|",
            )
            extract_context_attention(target_word, clustered_embeddings_df)
