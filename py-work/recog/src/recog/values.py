from pathlib import Path

from numpy import float64
from numpy.typing import NDArray

DATA = Path("../../data")


def coffee_path() -> str:
    return str(DATA / "coffee.jpg")


def cosmos_path() -> str:
    return str(DATA / "cosmos.jpg")
