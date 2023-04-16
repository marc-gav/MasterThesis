import yaml
import pickle
import re
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from loguru import logger
import os
from cluestar import plot_text

from marc_thesis.utils.file_manager import store_file_target, load_file_target
from marc_thesis.clustering_embeddings.cluster_embeddings import (
    cluster_embeddings,
)
from marc_thesis.clustering_embeddings.extract_bert_embeddings import (
    extract_bert_embeddings_of_word,
)

EXPERIMENTS_FOLDER = os.environ.get("MARC_THESIS_EXPERIMENT_FOLDER")


def main():
    target_word = input("Enter target word: ")
    cluster_method = input(
        "Which clustering method do you want to use? (DPBGMM, KMeans): "
    )
    number_clusters = int(input("How many clusters do you want to use? "))

    # get current file path
    current_file_path = os.path.dirname(os.path.abspath(__file__))
    with open(
        f"{current_file_path}/clustering_settings.yml", "r"
    ) as yaml_file:
        method_settings = yaml.safe_load(yaml_file)["methods"]

    sentences = load_file_target(
        target_word, f"{target_word}_sentences.pickle"
    )

    sentences_with_word = [sentence for idx, sentence in sentences.items()]
    indices_of_sentence_with_word = [
        idx for idx, sentence in sentences.items()
    ]
    sentences_with_word_dict = {
        "sentences": sentences_with_word,
        "sentence_idx": indices_of_sentence_with_word,
    }

    if not os.path.exists(
        f"{EXPERIMENTS_FOLDER}/{target_word}/{target_word}_embeddings.pickle"
    ):
        # 2. Extract the embeddings of the target word for each sentence
        embeddings_dict = extract_bert_embeddings_of_word(
            sentences_with_word_dict, target_word
        )
        store_file_target(
            target_word=target_word,
            file_name=f"{target_word}_embeddings.pickle",
            data=embeddings_dict,
        )
    else:
        embeddings_dict = load_file_target(
            target_word, f"{target_word}_embeddings.pickle"
        )

    embeddings = torch.stack(embeddings_dict["target_word_embeddings"]).numpy()

    # plot embeddings
    from sklearn.decomposition import PCA

    # 3. Cluster the embeddings
    cluster_labels = cluster_embeddings(
        embeddings, cluster_method, method_settings, number_clusters
    )

    cluster_labels_dict = {
        "cluster_labels": cluster_labels,
        "sentence_idx": embeddings_dict["sentence_idx"],
    }

    num_clusters = len(np.unique(cluster_labels))
    store_file_target(
        target_word=target_word,
        file_name=f"{target_word}_cluster_labels_{cluster_method}_{num_clusters}_clusters.pickle",
        data=cluster_labels_dict,
    )

    print("Clustering done. Plotting clusters...")
    # plot the clusters
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    pca.fit(embeddings.squeeze())
    embeddings_2d = pca.transform(embeddings.squeeze())

    plot_text(
        embeddings_2d,
        sentences_with_word,
        color_array=cluster_labels,
    ).show()

    tsne = TSNE(n_components=2, learning_rate="auto", init="pca")
    tsne.fit(embeddings.squeeze())

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(
        f"Clustering of {target_word} embeddings. {cluster_method} with {num_clusters} clusters"
    )
    # Assign colors to each cluster_label
    cluster_colors = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "purple", 5: "brown", 6: "pink", 7: "gray", 8: "olive", 9: "cyan"}
    cluster_labels = [cluster_colors[label] for label in cluster_labels]
    unique_clusters = np.unique(cluster_labels)
    ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels)
    ax1.set_title("PCA")
    ax2.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=cluster_labels)
    ax2.set_title("t-SNE")
    # add legend matching the color with the cluster label
    ax1.legend(
        [plt.Line2D((0, 1), (0, 0), color=color, linewidth=3, linestyle="-")
            for color in cluster_colors.values() if color in unique_clusters],
        [f"Cluster {cluster}" for cluster in cluster_colors.keys()],
    )
    ax2.legend(
        [plt.Line2D((0, 1), (0, 0), color=color, linewidth=3, linestyle="-")
            for color in cluster_colors.values() if color in unique_clusters],
        [f"Cluster {cluster}" for cluster in cluster_colors.keys()],
    )
    plt.savefig(
        f"{EXPERIMENTS_FOLDER}/../plots_for_thesis/Local_cluster_differences_xAI/{target_word}_{num_clusters}clusters_pca_tsne.pdf"
    )


if __name__ == "__main__":
    main()
