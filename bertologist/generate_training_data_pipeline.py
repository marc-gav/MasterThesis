import pandas as pd
from typing import List, Tuple
from bertologist.data.Datasets import (
    load_training_sentences,
)
from bertologist.extract_bert_embeddings import (
    get_embeddings_from_target_word_in_sentences,
)
from cluster_embeddings import cluster_embeddings
from extract_top_attention_words import extract_context_attention
from sklearn.decomposition import PCA

TARGET_WORD = "light"

print("Loading training sentences")
document_sentences: List[str] = load_training_sentences()
print(f"Training sentences have been loaded.")

print("Extracting embeddings")
embeddings, sentences = get_embeddings_from_target_word_in_sentences(
    document_sentences, TARGET_WORD
)
print(f"Embeddings have been extracted")

print("Clustering embeddings")
embedding_cluster_labels = cluster_embeddings(embeddings, method="DPBGMM")
print(f"Embeddings have been clustered")

print("PCA to embeddings")
pca = PCA(n_components=2)
pca.fit(embeddings)
embeddings_pca = pca.transform(embeddings)

df_sentence_clusters_attention = pd.DataFrame(
    columns=[
        "sentence",
        "word",
        "attention",
        "cluster_label",
        "sentence_index",
        "target_word",
        "pca1",
        "pca2",
    ]
)

df_sentence_clusters_attention["sentence"] = sentences
df_sentence_clusters_attention["cluster_label"] = embedding_cluster_labels
df_sentence_clusters_attention["sentence_index"] = range(len(sentences))
df_sentence_clusters_attention["target_word"] = TARGET_WORD

# for every different sentence_index assign the PCA values
for sentence_index in df_sentence_clusters_attention[
    "sentence_index"
].unique():
    df_sentence_clusters_attention.loc[
        df_sentence_clusters_attention["sentence_index"] == sentence_index,
        ["pca1", "pca2"],
    ] = embeddings_pca[sentence_index]


print("Extracting attention")
df_sentence_clusters_attention = extract_context_attention(
    TARGET_WORD, df_sentence_clusters_attention
)
print(f"Attention has been extracted")

with open("datasets/training_dataset.csv", "w") as f:
    df_sentence_clusters_attention.to_csv(f, index=False)
    print(f"Training dataset has been saved to {f.name}.")
