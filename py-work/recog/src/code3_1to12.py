import random

import numpy as np

from torch.utils.data import Dataset

import torchvision

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchvision.datasets.flowers102 import download_url


from recog.values import Image, NDArray


def main() -> None:
    dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True)

    displayed_classes = set()
    i = 0
    while i < len(dataset) and len(displayed_classes) < len(dataset.classes):
        img, label = dataset[i]
        if label not in displayed_classes:
            print(f"class: {dataset.classes[label]}")

            img = img.resize((256, 256))
            img.save(f"./outs/{i}.jpg")
            displayed_classes.add(label)
        i += 1

    # code 3.3
    x, y = [], []
    num_samples = 200
    for i in range(num_samples):
        img, label = dataset[i]

        img_flatten = np.asarray(img).flatten()
        x.append(img_flatten)
        y.append(label)

    x = np.stack(x)
    y = np.array(y)

    # code 3.4
    t_sne = TSNE(n_components=2, random_state=0)
    x_reduced = t_sne.fit_transform(x)

    cmap = plt.get_cmap("tab10")
    markers = ["4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(dataset.classes):
        plt.scatter(
            x_reduced[y == i, 0],
            x_reduced[y == i, 1],
            c=[cmap(i / len(dataset.classes))],
            marker=markers[i],
            s=500,
            alpha=0.6,
            label=cls,
        )
    plt.axis("off")
    plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
    plt.savefig("out.jpg")

    # code 3.6
    train_dataset = torchvision.datasets.CIFAR10(root="data", train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="data", train=False, download=True)

    val_ratio = 0.2

    val_set, train_set = generate_subset(train_dataset, val_ratio)

    print(f"train: {len(train_set)}")  # 40000
    print(f"value: {len(val_set)}")  # 10000
    print(f"test set: {len(test_dataset)}")  # 10000

    # code 3.11
    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform
    )

    channel_mean, channel_std = get_dataset_statistics(dataset)

    img_transform = lambda x: transform(x, channel_mean, channel_std)

    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=img_transform,
        target_transform=target_transform,
    )

    img, label = dataset[0]
    print(f"imag: {img}")
    print(f"label: {label}")  # [0,0,0,0,0,0,1,0,0,0]


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


def transform(img: Image, channel_mean: NDArray = None, channel_std: NDArray = None):
    img: NDArray[np.float32] = np.asarray(img, dtype="float32")

    x = img.flatten()

    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std
    return x


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


def target_transform(label: int, num_classes: int = 10):
    y = np.identity(num_classes)[label]
    return y


if __name__ == "__main__":
    main()
