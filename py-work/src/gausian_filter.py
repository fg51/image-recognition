"""\
smootihg filter
"""

import numpy as np

from PIL.Imagew import Image


def generate_gaussian_kernel(
    kernel_width: int, kernel_height: int, sigma: float
) -> np.ndarray:
    """\
    sigma: std dev of gauss for kernel
    """

    assert (kernel_width % 2 == 1) and (kernel_height % 2 == 1)

    kernel = np.empty((kernel_height, kernel_width))

    for y in range(-(kernel_height // 2), kernel_height // 2 + 1):
        for x in range(-(kernel_width // 2), kernel_width // 2 + 1):
            h = -(x * x + y * y) / (2.0 * sigma * sigma)
            h = np.exp(h) / (2.0 * np.pi * sigma * sigma)
            kernel[y + kernel_height // 2, x + kernel_width // 2] = h
    kernel /= np.sum(kernel)  # normalize : sum is 1
    return kernel


def convolution(img: Image.Image, kernel: np.ndarray, x: int, y: int):
    width, height = img.size
    kernel_height, kernel_width = kernel.shape[:2]

    value = 0
    for y_kernel in range(-(kernel_height // 2), kernel_height // 2 + 1):
        for x_kernel in range(-(kernel_width // 2), kernel_width // 2 + 1):
            x_img = max(min(x + x_kernel, width - 1), 0)
            y_img = max(min(y + y_kernel, height - 1), 0)
            h = kernel[y_kernel + kernel_height // 2, x_kernel + kernel_width // 2]
            value += h * img.getpixel((x_img, y_img))
    return value


def main():
    print(generate_gaussian_kernel(5, 5, 1.3))


if __name__ == "__main__":
    main()
