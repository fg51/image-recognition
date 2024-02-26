import random

import numpy as np

from torch.utils.data import Dataset


def generate_subset(dataset: Dataset, ratio: float, random_seed: int = 0):
    """code 3.5
    dataset: target
    ratio: amount of 1st set
    random_seed: seed
    """

    size = int(len(dataset) * ratio)
    indices = list(range(len(dataset)))

    random.seed(random_seed)
    random.shuffle(indices)
    indices1, indices2 = indices[:size], indices[size:]
    return indices1, indices2


def get_dataset_statistics(dataset: Dataset):
    # cocde 3.9
    data = []

    for i in range(len(dataset)):
        img_flat = dataset[i][0]
        data.append(img_flat)
    data = np.stack(data)

    channel_mean = np.mean(data, axis=0)
    channel_std = np.std(data, axis=0)

    return channel_mean, channel_std
