import argparse

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm


def parse_arguments(parser: argparse.ArgumentParser) -> argparse.Namespace:
    parser.add_argument(
        "--target_word",
        type=str,
        required=True,
        help="Target word to cluster",
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        required=False,
        help="Path to the csv file containing the embeddings and the context sentence",
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        required=False,
        default="DPBGMM",
        help="Clustering method to use",
    )
    parser.add_argument(
        "--clustering_method_settings_file",
        type=str,
        required=False,
        default="clustering_settings.yml",
        help="Path to the yaml file containing the settings for the clustering methods",
    )
    parser.add_argument(
        "--save_file",
        type=bool,
        required=False,
        default=True,
        help="Whether to save the cluster labels",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=False,
        default=f"data/{parser.parse_args().target_word}_embeddings_with_cluster_labels.csv",
        help="Path to save the cluster labels",
    )

    # Summarize the arguments
    print("Processing with arguments:")
    for arg in vars(parser.parse_args()):
        print(f"\t{arg}: {getattr(parser.parse_args(), arg)}")

    return parser.parse_args()


def cluster_embeddings(
    embeddings: np.ndarray, method: str, method_settings_file: str
) -> np.ndarray:
    """Cluster the embeddings using the specified method.

    Args:
        embeddings (np.ndarray): Embeddings to cluster.
        method (str): Clustering method to use.

    Returns:
        np.ndarray: Cluster labels.
    """

    # read from yaml file
    with open(method_settings_file, "r") as f:
        method_settings = yaml.safe_load(f)["methods"]

    if method == "DPBGMM":
        from sklearn.mixture import BayesianGaussianMixture

        dpbgmm_settings = method_settings["DPBGMM"]
        clustering = BayesianGaussianMixture(
            n_components=dpbgmm_settings["n_components"],
            weight_concentration_prior_type=dpbgmm_settings[
                "weight_concentration_prior_type"
            ],
            weight_concentration_prior=dpbgmm_settings[
                "weight_concentration_prior"
            ],
            covariance_type=dpbgmm_settings["covariance_type"],
            n_init=dpbgmm_settings["n_init"],
            max_iter=dpbgmm_settings["max_iter"],
            random_state=dpbgmm_settings["random_state"],
        )
        clustering.fit(embeddings)
        labels = clustering.predict(embeddings)
    elif method == "KMeans":
        from sklearn.cluster import KMeans

        kmeans_settings = method_settings["KMeans"]
        clustering = KMeans(
            n_clusters=kmeans_settings["n_clusters"],
            init=kmeans_settings["init"],
            n_init=kmeans_settings["n_init"],
            max_iter=kmeans_settings["max_iter"],
            random_state=kmeans_settings["random_state"],
        )
        clustering.fit(embeddings)
        labels = clustering.predict(embeddings)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    return labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cluster BERT embeddings and plot them in a 2D space using PCA."
    )
    args = parse_arguments(parser)
    target_word = parser.parse_args().target_word
    embeddings_file = parser.parse_args().embedding_file
    clustering_method = parser.parse_args().clustering_method
    clustering_method_settings_file = args.clustering_method_settings_file
    df_embeddings = pd.read_csv(embeddings_file, sep="|")
    df_embeddings["target_word_vector"] = df_embeddings[
        "target_word_vector"
    ].apply(lambda x: np.array(eval(x)))
    embeddings = np.stack(
        df_embeddings["target_word_vector"].to_numpy()
    ).squeeze()

    _target_word_senses = nltk.corpus.wordnet.synsets(target_word)
    target_word_senses_definitions = [
        sense.definition() for sense in _target_word_senses
    ]
    print(
        f"\nThe word {target_word} has a total of {len(target_word_senses_definitions)} senses."
    )
    for i, sense in enumerate(target_word_senses_definitions):
        print(f"\tSense {i+1}: {sense}")
    print()

    pca = PCA(n_components=2)
    pca_embeddings = pca.fit_transform(embeddings)
    labels = cluster_embeddings(
        embeddings,
        clustering_method,
        clustering_method_settings_file,
    )

    # save labels to csv file for further analysis
    df_clustered = df_embeddings.copy()
    df_clustered["cluster_label"] = labels
    df_clustered["clustering_method"] = clustering_method
    df_clustered["target_word"] = target_word
    # drop target_word_vector column
    df_clustered.drop(columns=["target_word_vector"], inplace=True)
    df_clustered.to_csv(
        f"{args.save_path}",
        sep="|",
        index=False,
    )
    print(f"Cluster labels saved to {args.save_path}")

    fig, ax = plt.subplots()
    ax.scatter(
        pca_embeddings[:, 0],
        pca_embeddings[:, 1],
        alpha=0.9,
        c=labels,
        cmap="tab10",
    )
    ax.set_title("Bert embeddings of word 'plant' in sentence")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_axisbelow(False)
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.show()
