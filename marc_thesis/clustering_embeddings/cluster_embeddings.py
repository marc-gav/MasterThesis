import numpy as np
import pickle
import yaml
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
from loguru import logger
from cluestar import plot_text

from marc_thesis.utils.file_manager import store_file_target, load_file_target


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str,
    method_settings: dict,
    number_clusters: int,
) -> np.ndarray:

    logger.info(f"Clustering embeddings with {method}...")
    if method == "DPBGMM":
        logger.info(f"Ignoring number_clusters parameter for {method}...")
        from sklearn.mixture import BayesianGaussianMixture

        # clustering using dirichlet process mixture model. Number of cluster is
        # determined by the algorithm
        clustering = BayesianGaussianMixture(
            n_components=number_clusters,
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=100,
            n_init=10,
            init_params="k-means++",
            random_state=42,
        )

        clustering.fit(embeddings.squeeze())
        labels = clustering.predict(embeddings.squeeze())
    elif method == "KMeans":
        from sklearn.cluster import KMeans

        kmeans_settings = method_settings["KMeans"]
        clustering = KMeans(
            n_clusters=number_clusters,
            init=kmeans_settings["init"],
            n_init=kmeans_settings["n_init"],
            max_iter=kmeans_settings["max_iter"],
            random_state=kmeans_settings["random_state"],
        )
        clustering.fit(embeddings.squeeze())
        labels = clustering.predict(embeddings.squeeze())
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels


def main():
    target_word = input("Target word: ")
    cluster_method = input(
        "Which clustering method do you want to use? (DPBGMM, KMeans): "
    )
    embeddings_dict = load_file_target(
        target_word, f"{target_word}_embeddings.pickle"
    )

    sentence_idx = embeddings_dict["sentence_idx"]
    embeddings = torch.stack(embeddings_dict["target_word_embeddings"])

    yaml_file = open("clustering_settings.yml", "r")
    cluster_settings = yaml.load(yaml_file, Loader=yaml.FullLoader)
    labels = cluster_embeddings(
        embeddings, cluster_method, cluster_settings["methods"]
    )

    cluster_labels_dict = {
        "cluster_labels": labels,
        "sentence_idx": sentence_idx,
    }

    num_clusters = len(np.unique(labels))

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

    tsne = TSNE(n_components=2, learning_rate="auto", init="pca")
    tsne.fit(embeddings.squeeze())

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels)
    ax1.set_title("PCA")
    ax2.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=labels)
    ax2.set_title("t-SNE")
    plt.show()

    # Save the plot to a file
    fig.savefig(
        f"../experiment_files/{target_word}/{target_word}_cluster_labels_{cluster_method}_{num_clusters}_clusters.png"
    )


if __name__ == "__main__":
    main()
