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
    channel_mean, channel_std = util.get_dataset_statistics(dataset)

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

    model = VisionTransformer(
        len(train_dataset.classes),
        32,
        config.patch_size,
        config.dim_hidden,
        config.num_heads,
        config.dim_feedforward,
        config.num_layers,
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

        val_loss, val_accuracy = evaluate.evaluate(val_loader, model, loss_func)
        print(f"evacuate: loss = {val_loss:.3f}, accuracy = {val_accuracy:.3f}")

        if val_loss < val_loss_best:
            val_loss_best = val_loss
            model_best = model.copy()

    test_loss, test_accuracy = evaluate.evaluate(test_loader, model_best, loss_func)
    print(f"test: loss = {test_loss:.3f}, accuracy = {test_accuracy:.3f}")

    plot_t_sne("out-4-21.png", test_loader, model_best, config.num_samples)


class SelfAttention(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()

        assert dim_hidden % num_heads == 0

        self.num_heads = num_heads
        dim_head = dim_hidden // num_heads
        self.scale = dim_head**-0.5
        self.proj_in = nn.Linear(dim_hidden, dim_hidden * 3, bias=qkv_bias)
        self.proj_out = nn.Linear(dim_hidden, dim_hidden)

    def forward(self, x: Tensor):
        bs, ns = x.shape[:2]

        qkv = self.proj_in(x)

        qkv = qkv.view(bs, ns, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = q.matmul(k.transpose(-2, -1))
        attn = (attn * self.scale).softmax(dim=-1)

        x = attn.matmul(v)

        x = x.permute(0, 2, 1, 3).flatten(2)
        x = self.proj_out(x)
        return x


class FNN(nn.Module):
    def __init__(self, dim_hidden: int, dim_feedforward: int):
        super().__init__()
        self.linear1 = nn.Linear(dim_hidden, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_hidden)
        self.activation = nn.GELU()

    def forward(self, x: Tensor):
        return self.linear2(self.activation(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_hidden: int, num_heads: int, dim_feedforward: int):
        super().__init__()
        self.attention = SelfAttention(dim_hidden, num_heads)
        self.fnn = FNN(dim_hidden, dim_feedforward)

        self.norm1 = nn.LayerNorm(dim_hidden)
        self.norm2 = nn.LayerNorm(dim_hidden)

    def forward(self, x: Tensor):
        x = self.norm1(x)
        x = self.attention(x) + x
        x = self.norm2(x)
        return self.fnn(x) + x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_size: int,
        patch_size: int,
        dim_hidden: int,
        num_heads: int,
        dim_feedforward: int,
        num_layers: int,
    ):
        super().__init__()

        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size

        num_patches = (img_size // patch_size) ** 2

        dim_patch = 3 * patch_size**2

        self.patch_embed = nn.Linear(dim_patch, dim_hidden)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim_hidden))

        self.class_token = nn.Parameter(torch.zeros((1, 1, dim_hidden)))

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_hidden, num_heads, dim_feedforward)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(dim_hidden)
        self.linear = nn.Linear(dim_hidden, num_classes)

    def forward(self, x: Tensor, return_embed: bool = False):
        bs, c, h, w = x.shape

        assert h == self.img_size and w == self.img_size

        x = x.view(
            bs,
            c,
            h // self.patch_size,
            self.patch_size,
            w // self.patch_size,
            self.patch_size,
        )

        x = x.permute(0, 2, 4, 1, 3, 5)

        x = x.reshape(bs, (h // self.patch_size) * (w // self.patch_size), -1)
        x = self.patch_embed(x)

        class_token = self.class_token.expand(bs, -1, -1)
        x = torch.cat((class_token, x), dim=1)
        x += self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = x[:, 0]

        x = self.norm(x)
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
    patch_size = 4
    dim_hidden = 512
    num_heads = 8
    dim_feedforward = 512
    num_layers = 6
    num_epochs = 30
    lr = 1e-2
    moving_avg = 20
    batch_size = 32
    num_workers = 2
    device = "cpu"
    num_samples = 200


def plot_t_sne(filepath: str, data_loader: Dataset, model: nn.Module, num_samples: int):
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
        plt.savefig(filepath)


if __name__ == "__main__":
    main()
