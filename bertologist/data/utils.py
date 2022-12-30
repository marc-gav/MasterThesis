import torch
from bertologist.data.Datasets import ClusteredWordsDataset


def split_dataset(dataset, split_values: list):
    """Splits the dataset into multiple datasets based on the split_values.
    The split_values should be a list of floats that sum to 1.
    """

    assert sum(split_values) == 1, "The split values should sum to 1"
    assert len(split_values) > 1, "There should be at least 2 split values"

    dataset_sizes = [
        int(len(dataset) * split_value) for split_value in split_values
    ]
    leftover = len(dataset) - sum(dataset_sizes)
    dataset_sizes[0] += leftover

    # Split dataset.data accordingly
    data_splits = torch.split(dataset.data, dataset_sizes)
    label_splits = torch.split(dataset.labels, dataset_sizes)
    datasets = []
    for data, labels in zip(data_splits, label_splits):
        datasets.append(ClusteredWordsDataset(data=data, labels=labels))

    return datasets
