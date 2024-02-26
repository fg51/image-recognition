import numpy as np

from recog.values import Image, NDArray


def transform(img: Image, channel_mean: NDArray = None, channel_std: NDArray = None):
    img: NDArray[np.float32] = np.asarray(img, dtype="float32")

    x = img.flatten()

    if channel_mean is not None and channel_std is not None:
        x = (x - channel_mean) / channel_std
    return x
