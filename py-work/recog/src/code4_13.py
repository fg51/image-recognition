import copy
from collections import deque
from dataclasses import dataclass

from tqdm import tqdm

import torch
from torch import nn, optim, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision import transforms as T

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

from recog import util, evaluate


def main() -> None:
    config = Config()

    dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=T.ToTensor()
    )
    channel_mean, channel_std = get_dataset_statistics(dataset)

    transforms = T.Compose(
        (T.ToTensor(), T.Normalize(mean=channel_mean, std=channel_std))
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="data", train=True, download=True, transform=transforms
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="data", train=False, download=True, transform=transforms
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

    model = ResNet18(len(train_dataset.classes))

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

        val_loss, val_accuracy = evaluate.evaluate(val_loader, model, loss_func)
        print(f"evacuate: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()

    test_loss, test_accuracy = evaluate.evaluate(test_loader, model_best, loss_func)
    print(f"test: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}")

    plot_t_sne(test_loader, model_best, config.num_samples)


def get_dataset_statistics(dataset: Dataset):
    # cocde 4.13
    data = []

    for i in range(len(dataset)):
        img = dataset[i][0]
        data.append(img)
    data = torch.stack(data)

    channel_mean = data.mean(dim=(0, 2, 3))
    channel_std = data.std(dim=(0, 2, 3))

    return channel_mean, channel_std


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


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor):
        out = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))

        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2), BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2), BasicBlock(512, 512)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512, num_classes)

    def forward(self, x: Tensor, return_embed: bool = False):
        x = self.max_pool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))

        x = self.avg_pool(x)
        x = x.flatten(1)

        if return_embed:
            return x
        x = self.linear(x)
        return x

    def get_device(self):
        return self.linear.weight.device

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class Config:
    val_ratio = 0.2
    num_epochs = 30
    lr = 1e-2
    moving_avg = 20
    batch_size = 32
    num_workers = 2
    device = "cpu"
    num_samples = 200


if __name__ == "__main__":
    main()
