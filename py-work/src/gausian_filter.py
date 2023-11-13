"""\
smootihg filter
"""

import numpy as np

from PIL import Image


def generate_gaussian_kernel(
    kernel_width: int, kernel_height: int, sigma: float
) -> np.ndarray:
    """\
    sigma: std dev of gauss for kernel
    """

    width_half, height_half = kernel_width // 2, kernel_height // 2
    assert (width_half == 1) and (height_half == 1)

    kernel = np.empty((kernel_height, kernel_width))

    for y in range(-height_half, height_half + 1):
        for x in range(-width_half, width_half + 1):
            h = -(x * x + y * y) / (2.0 * sigma * sigma)
            h = np.exp(h) / (2.0 * np.pi * sigma * sigma)
            kernel[y + height_half, x + width_half] = h
    kernel /= np.sum(kernel)  # normalize : sum is 1
    return kernel


def convolution(img: Image.Image, kernel: np.ndarray, x: int, y: int):
    width, height = img.size
    kernel_height, kernel_width = kernel.shape[:2]

    width_half, height_half = kernel_width // 2, kernel_height // 2
    value = 0
    for y_kernel in range(-height_half, height_half + 1):
        for x_kernel in range(-width_half, width_half + 1):
            x_img = max(min(x + x_kernel, width - 1), 0)
            y_img = max(min(y + y_kernel, height - 1), 0)
            h = kernel[y_kernel + height_half, x_kernel + width_half]
            value += h * img.getpixel((x_img, y_img))
    return value


def apply_filter(img: Image.Image, kernel: np.ndarray):
    width, height = img.size

    img_filtered = Image.new(mode="L", size=(width, height))

    for y in range(height):
        for x in range(width):
            filtered_value = convolution(img, kernel, x, y)
            img_filtered.putpixel((x, y), int(filtered_value))
    return img_filtered


def main():
    print(generate_gaussian_kernel(5, 5, 1.3))


if __name__ == "__main__":
    main()
