from bertologist.utils import get_top_n_attention_words_to_target
from tqdm import tqdm
import os
import re
import pandas as pd


def extract_context_attention(
    target_word: str,
    clustered_embeddings_df: pd.DataFrame,
):
    """Extract the attention weights of the context words towards the target word.

    Args:
        target_word (str): Target word
        clustered_embeddings_df (pd.DataFrame): Dataframe of clustered embeddings
        store_csv (bool, optional): Store the attention weights in a csv file. Defaults to True.

    Returns:
        # TODO: define return type
    """

    new_df = pd.DataFrame(columns=clustered_embeddings_df.columns)
    # for each sentence_index in the dataframe, get the attention weights
    # of the context words towards the target word
    for sentence_index in tqdm(
        clustered_embeddings_df["sentence_index"].unique()
    ):
        sentence = clustered_embeddings_df[
            clustered_embeddings_df["sentence_index"] == sentence_index
        ]["sentence"].values[0]
        for top_n_attention in get_top_n_attention_words_to_target(
            sentence, target_word, n=10  # Top 10 words
        ):

            # add each word, attention as a row for the dataframe
            for word, score in top_n_attention:
                # with that sentence index, add a new row to the dataframe
                row = clustered_embeddings_df.loc[
                    clustered_embeddings_df["sentence_index"] == sentence_index
                ].iloc[0]
                row["word"] = word
                row["salience_value"] = score
                row["salience_score"] = "attention"
                # concat
                new_df = pd.concat([new_df, row.to_frame().T])

    return new_df


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
