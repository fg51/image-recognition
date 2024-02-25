from recog.values import cosmos_path

from PIL import Image

import numpy as np


def main() -> None:
    img = Image.open(cosmos_path())
    img = np.asarray(img, dtype="float32")

    w = np.array(
        [
            [0.0065, -0.0045, -0.0018, 0.0075, 0.0095, 0.0075, -0.0026, 0.0022],
            [-0.0065, 0.0081, 0.0097, -0.0070, -0.0086, -0.0107, 0.0062, -0.0050],
            [0.0024, -0.0018, 0.0002, 0.0023, 0.0017, 0.0021, -0.0017, 0.0016],
        ]
    )
    features = np.matmul(img, w)

    # code 2.21
    feature_white = features[50, 50]
    feature_pink = features[200, 200]

    atten_white = np.matmul(features, feature_white)
    atten_pink = np.matmul(features, feature_pink)

    atten_white = np.exp(atten_white) / np.sum(np.exp(atten_white))
    atten_pink = np.exp(atten_pink) / np.sum(np.exp(atten_pink))

    # code 2.22
    atten_white = (atten_white - np.amin(atten_white)) / (
        np.amax(atten_white) - np.amin(atten_white)
    )
    atten_pink = (atten_pink - np.amin(atten_pink)) / (
        np.amax(atten_pink) - np.amin(atten_pink)
    )

    img_atten_white = Image.fromarray((atten_white * 255).astype("uint8"))
    img_atten_pink = Image.fromarray((atten_pink * 255).astype("uint8"))

    img_atten_white.save("img_atten_white.jpg")
    img_atten_pink.save("img_atten_pink.jpg")


if __name__ == "__main__":
    main()
