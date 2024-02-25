from PIL import Image
import numpy as np

from recog.values import NDArray, float64


def main() -> None:
    """code 2.10, 2.12"""
    kernel = generate_gaussian_kernel(kernel_width=5, kernel_height=5, sigma=1.3)
    print("gaussian kernel")
    print(kernel)

    img = Image.open("../../data/coffee_noise.jpg")
    # img.show()
    img_filtered = apply_fillter(img, kernel)
    img_filtered.save("coffee_noise_filtered_out.jpg")
    # img_filtered.show()


def code2_1() -> None:
    img_gray = Image.open("coffee.jpg")
    print("{}".format(np.array(img_gray).shape))
    print("{}".format(img_gray.getpixel((0, 0))))


def generate_gaussian_kernel(
    kernel_width: int, kernel_height: int, sigma: float
) -> NDArray[float64]:
    """code:2.7
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


def convolution(img: Image.Image, kernel: NDArray[float64], x: int, y: int) -> float:
    """code 2.8"""
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


def apply_fillter(img: Image.Image, kernel: NDArray[float64]) -> Image.Image:
    """code:2.9"""
    width, height = img.size
    img_filtered = Image.new(mode="L", size=(width, height))

    for y in range(height):
        for x in range(width):
            filtered_value = convolution(img, kernel, x, y)
            img_filtered.putpixel((x, y), int(filtered_value))
    return img_filtered


if __name__ == "__main__":
    main()
