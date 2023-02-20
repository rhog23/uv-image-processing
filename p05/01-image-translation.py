import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt


def run(image="", tx=0, ty=0):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transformation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

    result = cv2.warpAffine(img, transformation_matrix, (img.shape[1], img.shape[0]))

    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(122)
    plt.imshow(result)
    plt.axis("off")
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path to image", required=True)
    parser.add_argument(
        "--tx",
        type=int,
        required=True,
        help="horizontal value for affine transformation matrix, can be positive or negative",
    )
    parser.add_argument(
        "--ty",
        type=int,
        required=True,
        help="vertical value for affine transformation matrix, can be positive or negative",
    )
    opt = parser.parse_args()
    print(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
