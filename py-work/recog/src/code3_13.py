from collections import deque

import numpy as np

from tqdm import tqdm

import torchvision

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from recog.values import NDArray

from code3_1to12 import (
    get_dataset_statistics,
    target_transform,
    transform,
    generate_subset,
)


def main() -> None:
    model = MultiClassLogisticRegression(32 * 32 * 3, 10)
    x = np.random.normal(size=(1, 32 * 32 * 3))
    y = model.predict(x)
    print(f"predict: {y[0]}")

    config = Config()

    dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    channel_mean, channel_std = get_dataset_statistics(dataset)

    img_transform = lambda x: transform(x, channel_mean, channel_std)

    train_dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=img_transform,
        target_transform=target_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=img_transform,
        target_transform=target_transform,
    )

    val_set, train_set = generate_subset(train_dataset, config.val_ratio)

    print(f"{len(train_set)}")
    print(f"{len(val_set)}")
    print(f"{len(test_dataset)}")

    train_sampler = SubsetRandomSampler(train_set)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=train_sampler,
    )

    val_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=val_set,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    val_loss_best = float("inf")
    model_best = None

    for lr in config.lrs:
        print(f"lean ratio: {lr}")

        model = MultiClassLogisticRegression(32 * 32 * 3, len(train_dataset.classes))

        for epoch in range(config.num_epochs):
            with tqdm(train_loader) as pbar:
                pbar.set_description(f"[epoch {epoch + 1}]")

                losses = deque()
                accs = deque()

                for x, y in pbar:
                    x = x.numpy()
                    y = y.numpy()

                    y_pred = model.predict(x)

                    loss = np.mean(np.sum(-y * np.log(y_pred), axis=1))

                    accuracy = np.mean(
                        np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
                    )

                    losses.append(loss)
                    accs.append(accuracy)

                    if len(losses) > config.moving_avg:
                        losses.popleft()
                        accs.popleft()
                    pbar.set_postfix(
                        {"loss": np.mean(losses), "accuracy": np.mean(accs)}
                    )

                    model.update_parameters(x, y, y_pred, lr=lr)

            val_loss, val_accuracy = evaluate(val_loader, model)

            print(f"検証: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")

            if val_loss < val_loss_best:
                val_loss_best = val_loss
                model_best = model.copy()
    test_loss, test_accuracy = evaluate(test_loader, model_best)
    print(f"test: loss = {test_loss:.3f},")
    print(f"accuracy = {test_accuracy:.3f},")


def evaluate(data_loader, model):
    losses = []
    preds = []

    for x, y in data_loader:
        x = x.numpy()
        y = y.numpy()
        y_pred = model.predict(x)

        losses.append(np.sum(-y * np.log(y_pred), axis=1))
        preds.append(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))
    loss = np.mean(np.concatenate(losses))
    accuracy = np.mean(np.concatenate(preds))
    return loss, accuracy


class MultiClassLogisticRegression:
    def __init__(self, dim_input: int, num_classes: int) -> None:
        self.weight = np.random.normal(scale=0.01, size=(dim_input, num_classes))

        self.bias = np.zeros(num_classes)

    def _softmax(self, x: NDArray):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def predict(self, x: NDArray):
        l = np.matmul(x, self.weight) + self.bias
        y = self._softmax(l)
        return y

    def update_parameters(
        self, x: NDArray, y: NDArray, y_pred: NDArray, lr: float = 0.001
    ):
        diffs = y_pred - y

        self.weight -= lr * np.mean(x[:, :, np.newaxis] * diffs[:, np.newaxis], axis=0)
        self.bias -= lr * np.mean(diffs, axis=0)

    def copy(self):
        model_copy = self.__class__(*self.weight.shape)
        model_copy.weight = self.weight.copy()
        model_copy.bias = self.bias.copy()
        return model_copy


class Config:
    def __init__(self):
        self.val_ratio = 0.2
        self.num_epochs = 30
        self.lrs = [1e-2, 1e-3, 1e-4]
        self.moving_avg = 20
        self.batch_size = 32
        self.num_workers = 2


if __name__ == "__main__":
    main()
