import os
import pickle
from loguru import logger

# read $MARC_THESIS_EXPERIMENT_FOLDER from environment
assert (
    "MARC_THESIS_EXPERIMENT_FOLDER" in os.environ
), """Environment variable MARC_THESIS_EXPERIMENT_FOLDER not set.
It is probably 'MasterThesis/experiment_files'
Please set it to the path where you want to store the experiment files such as
the generated sentences, the embeddings, the clusters, etc."""

FILES_ROOT = os.environ.get("MARC_THESIS_EXPERIMENT_FOLDER")


def store_all_sentences(pickle_name: str, data):
    with open(f"{FILES_ROOT}/{pickle_name}", "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Stored {FILES_ROOT}/{pickle_name}")


def store_file_target(target_word: str, file_name: str, data):
    dir = f"{FILES_ROOT}/{target_word}"
    if not os.path.exists(dir):
        os.makedirs(dir)
        logger.info(f"Created directory {dir}")

    with open(f"{dir}/{file_name}", "wb") as f:
        pickle.dump(data, f)
        logger.info(f"Stored {dir}/{file_name}")


def load_file_target(target_word: str, file_name: str):
    with open(f"{FILES_ROOT}/{target_word}/{file_name}", "rb") as f:
        return pickle.load(f)
