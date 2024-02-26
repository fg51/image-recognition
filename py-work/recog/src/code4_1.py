from dataclasses import dataclass
from collections import deque
import copy
from typing import Callable

from tqdm import tqdm

import torch
from torch import nn, optim, Tensor
import torchvision

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt


from recog import transform, util


def main() -> None:
    config = Config()

    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transform.transform
    )
    channel_mean, channel_std = util.get_dataset_statistics(dataset)

    img_transform = lambda x: transform.transform(x, channel_mean, channel_std)

    train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=img_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=img_transform
    )

    val_set, train_set = util.generate_subset(train_dataset, config.val_ratio)

    print(f"lean set samples: {len(train_set)}")
    print(f"eval set samples: {len(val_set)}")
    print(f"test set samples: {len(test_dataset)}")

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

    loss_func = F.cross_entropy

    val_loss_best = float("inf")
    model_best = None

    model = FNN(
        32 * 32 * 3,
        config.dim_hidden,
        config.num_hidden_layers,
        len(train_dataset.classes),
    )

    model.to(config.device)

    optimizer = optim.SGD(model.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):
        model.train()

        with tqdm(train_loader) as pbar:
            pbar.set_description(f"[epoch {epoch + 1}]")

            losses = deque()
            accs = deque()
            for x, y in pbar:
                x = x.to(model.get_device())
                y = y.to(model.get_device())

                optimizer.zero_grad()

                y_pred = model(x)

                loss = loss_func(y_pred, y)
                accuracy = (y_pred.argmax(dim=1) == y).float().mean()

                loss.backward()

                optimizer.step()

                losses.append(loss.item())
                accs.append(accuracy.item())

                if len(losses) > config.moving_avg:
                    losses.popleft()
                    accs.popleft()
                pbar.set_postfix(
                    {
                        "loss": Tensor(losses).mean().item(),
                        "accuracy": Tensor(accs).mean().item(),
                    }
                )
        val_loss, val_accuracy = evaluate(val_loader, model, loss_func)
        print(f"evacuate: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()

    test_loss, test_accuracy = evaluate(test_loader, model_best, loss_func)
    print(f"test: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}")

    plot_t_sne(test_loader, model_best, config.num_samples)


def evaluate(data_loader: Dataset, model: nn.Module, loss_func: Callable):
    model.eval()
    losses = []
    preds = []
    for x, y in data_loader:
        with torch.no_grad():
            x = x.to(model.get_device())
            y = y.to(model.get_device())
            y_pred = model(x)

            losses.append(loss_func(y_pred, y, reduction="none"))
            preds.append(y_pred.argmax(dim=1) == y)
    loss = torch.cat(losses).mean()
    accuracy = torch.cat(preds).float().mean()
    return loss, accuracy


def plot_t_sne(data_loader: Dataset, model: nn.Module, num_samples: int):
    model.eval()

    x, y = [], []
    for imgs, labels in data_loader:
        with torch.no_grad():
            imgs = imgs.to(model.get_device())
            embeddings = model(imgs, return_embed=True)
            x.append(embeddings.to("cpu"))
            y.append(labels.clone())
    x = torch.cat(x)
    y = torch.cat(y)
    x = x.numpy()
    y = y.numpy()

    x = x[:num_samples]
    y = y[:num_samples]

    t_sne = TSNE(n_components=2, random_state=0)
    x_reduced = t_sne.fit_transform(x)

    cmap = plt.get_cmap("tab10")
    markers = ["4", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    plt.figure(figsize=(20, 15))
    for i, cls in enumerate(data_loader.dataset.classes):
        plt.scatter(
            x_reduced[y == i, 0],
            x_reduced[y == i, 1],
            c=[cmap(i / len(data_loader.dataset.classes))],
            marker=markers[i],
            s=500,
            alpha=0.6,
            label=cls,
        )
        plt.axis("off")
        plt.legend(bbox_to_anchor=(1, 1), fontsize=24, framealpha=0)
        plt.savefig("out.png")


class FNN(nn.Module):
    def __init__(
        self, dim_input: int, dim_hidden: int, num_hidden_layers: int, num_classes: int
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(self._generate_hidden_layer(dim_input, dim_hidden))
        for _ in range(num_hidden_layers - 1):
            self.layers.append(self._generate_hidden_layer(dim_hidden, dim_hidden))
        self.linear = nn.Linear(dim_hidden, num_classes)

    def _generate_hidden_layer(self, dim_input: int, dim_output: int):
        layer = nn.Sequential(
            nn.Linear(dim_input, dim_output, bias=False),
            nn.BatchNorm1d(dim_output),
            nn.ReLU(inplace=True),
        )
        return layer

    def forward(self, x: Tensor, return_embed: bool = False):
        h = x
        for layer in self.layers:
            h = layer(h)

            if return_embed:
                return h

            y = self.linear(h)
            return y

    def get_device(self):
        return self.linear.weight.device

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class Config:
    val_ratio = 0.2
    dim_hidden = 512
    num_hidden_layers = 2
    num_epochs = 30
    lr = 1e-2
    moving_avg = 20
    batch_size = 32
    num_workers = 2
    device = "cpu"
    num_samples = 200


if __name__ == "__main__":
    main()
