import argparse
import cv2
import matplotlib.pyplot as plt


def run(image="", flip=-1):
    flip_code = {0, 1, -1}
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if flip not in flip_code:
        print("Invalid flip code")
        return False

    result = cv2.flip(img, flip)

    show_result(img, result)


def show_result(orig_img, result):
    plt.figure()
    plt.subplot(121)
    plt.title("Original Image", fontsize=24)
    plt.axis("off")
    plt.imshow(orig_img)
    plt.subplot(122)
    plt.title("Result", fontsize=24)
    plt.axis("off")
    plt.imshow(result)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to image")
    parser.add_argument(
        "-f",
        "--flip",
        type=int,
        default=-1,
        help="flipping code:\n1: flip horizontally\n0:flip vertically\n-1: flip horizontally and vertically",
    )
    opt = parser.parse_args()
    print(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
