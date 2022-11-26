# For every sentence in the document
# Extract embedding of target word
# Cluster embeddings of context words
# Extract top 10 attention weights towards target word

# Save as a csv file the following columns
# sentence, word, attention, cluster_label, sentence_index


from typing import Tuple
import json

import nltk
import pandas as pd
import torch

from bertologist.utils import (
    split_into_sentences,
)
from bertologist.data.Datasets import SentencesDataset
from bertologist.extract_bert_embeddings import (
    get_embeddings_from_target_word_in_sentences,
)
from cluster_embeddings import cluster_embeddings
from extract_top_attention_words import extract_context_attention

TARGET_WORD = "light"
document_sentences: list[str] = SentencesDataset().sentences
sentence_embedding_tuples = get_embeddings_from_target_word_in_sentences(
    document_sentences, TARGET_WORD, store_csv=False
)
embeddings = torch.stack(
    [embedding for _, embedding in sentence_embedding_tuples]
).squeeze()
sentences = [sentence for sentence, _ in sentence_embedding_tuples]

embedding_cluster_labels = cluster_embeddings(embeddings, method="DPBGMM")

df_sentence_clusters_attention = pd.DataFrame(
    columns=[
        "sentence",
        "cluster_label",
        "sentence_index",
    ]
)

df_sentence_clusters_attention["sentence"] = sentences
df_sentence_clusters_attention["cluster_label"] = embedding_cluster_labels
df_sentence_clusters_attention["sentence_index"] = range(len(sentences))
df_sentence_clusters_attention = extract_context_attention(
    TARGET_WORD, df_sentence_clusters_attention, store_csv=False
)

with open("data/training_dataset.csv", "w") as f:
    df_sentence_clusters_attention.to_csv(f, index=False)
