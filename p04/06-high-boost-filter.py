import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2


def run(constants, image=""):
    if len(constants) < 1:
        return "Please input at least one constant"

    if len(constants) > 3:
        return "Please input not more than three constants"

    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    kernels = [
        np.array([[-1, -1, -1], [-1, c, -1], [-1, -1, -1]], dtype=np.float32)
        for c in constants
    ]

    results = [cv2.filter2D(img, -1, k) for k in kernels]

    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    axs[0, 0].set_title("Original", fontsize=16)
    axs[0, 0].set_axis_off()
    axs[0, 0].imshow(img)
    axs[0, 1].set_title(f"Result | c={constants[0]}", fontsize=16)
    axs[0, 1].set_axis_off()
    axs[0, 1].imshow(results[0])
    axs[1, 0].set_title(f"Result | c={constants[1]}", fontsize=16)
    axs[1, 0].set_axis_off()
    axs[1, 0].imshow(results[1])
    axs[1, 1].set_title(f"Result | c={constants[2]}", fontsize=16)
    axs[1, 1].set_axis_off()
    axs[1, 1].imshow(results[2])
    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to image")
    parser.add_argument(
        "-c",
        "--constants",
        default=[9, 10, 11],
        nargs="+",
        help="set of constants. minimal 1, maximum 3",
    )
    opt = parser.parse_args()
    print(vars(opt))
    return opt


def main(opt):
    try:
        run(**vars(opt))
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
