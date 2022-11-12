import pandas as pd
from bert_utils import get_top_n_attention_words_to_target
from tqdm import tqdm

if __name__ == "__main__":
    df_sentences_clustered = pd.read_csv(
        "data/plant_embeddings_with_cluster_labels.csv",
        sep="|",
        index_col=False,
    )

    # Remove repeated sentences
    df_sentences_clustered.drop_duplicates(subset=["sentence"], inplace=True)

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
        df_sentences_clustered.iterrows(),
        desc="Extracting top attention words",
        total=len(df_sentences_clustered),
    ):
        sentence = row["sentence"]
        for top_n_attention in get_top_n_attention_words_to_target(
            sentence, "plant", n=5
        ):

            top_n_attention["sentence_index"] = sentence_number
            top_n_attention["salience score"] = "attention"
            top_n_attention["sentence"] = sentence
            top_n_attention["cluster_label"] = row["cluster_label"]

            df_top_attention_all_sentences = pd.concat(
                [
                    df_top_attention_all_sentences,
                    top_n_attention,
                ]
            )

            sentence_number += 1

    df_top_attention_all_sentences.to_csv(
        "data/plant_top_attention_words.csv",
        sep="|",
        index=False,
    )
    print(f'Top attention words saved to "data/plant_top_attention_words.csv"')
