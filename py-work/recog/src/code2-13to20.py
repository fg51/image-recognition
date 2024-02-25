from PIL import Image
import numpy as np
from scipy import signal

from recog.values import NDArray
from recog.values import coffee_path


def main() -> None:
    """code 2.16 to 2.20"""
    kernel_h, kernel_v, kernel_lap = generate_kernel()
    print(kernel_h)
    print(kernel_v)
    print(kernel_lap)

    img = Image.open(coffee_path())
    img = np.asarray(img, dtype="int32")

    img_h_diff = signal.convolve2d(img, kernel_h, mode="same")
    img_v_diff = signal.convolve2d(img, kernel_v, mode="same")
    img_lap = signal.convolve2d(img, kernel_lap, mode="same")

    img_h_diff = np.absolute(img_h_diff)
    img_v_diff = np.absolute(img_v_diff)

    img_diff = (img_h_diff**2 + img_v_diff**2) ** 0.5

    img_h_diff = np.clip(img_h_diff, 0, 255).astype("uint8")
    img_v_diff = np.clip(img_v_diff, 0, 255).astype("uint8")
    img_diff = np.clip(img_diff, 0, 255).astype("uint8")
    img_lap = np.clip(img_lap, 0, 255).astype("uint8")

    img_h_diff = Image.fromarray(img_h_diff)
    img_v_diff = Image.fromarray(img_v_diff)
    img_diff = Image.fromarray(img_diff)
    img_lap = Image.fromarray(img_lap)

    img_h_diff.save("img_h_diff.jpg")
    img_v_diff.save("img_v_diff.jpg")
    img_diff.save("img_diff.jpg")
    img_lap.save("img_lap.jpg")


def generate_kernel() -> tuple[NDArray, NDArray, NDArray]:
    """code 2.14"""
    kernel_h = np.zeros((3, 3))
    kernel_v = np.zeros((3, 3))
    kernel_lap = np.zeros((3, 3))

    kernel_h[1, 1] = -1
    kernel_h[1, 2] = 1

    kernel_v[1, 1] = -1
    kernel_v[2, 1] = 1

    kernel_lap[0, 1] = 1
    kernel_lap[1, 0] = 1
    kernel_lap[1, 2] = 1
    kernel_lap[2, 1] = 1
    kernel_lap[1, 1] = -4

    return kernel_h, kernel_v, kernel_lap


if __name__ == "__main__":
    main()
