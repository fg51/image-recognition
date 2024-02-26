from typing import Callable
import torch
from torch import nn
from torch.utils.data import Dataset


def evaluate(data_loader: Dataset, model: nn.Module, loss_func: Callable):
    model.eval()

    losses, preds = [], []

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
