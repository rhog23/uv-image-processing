import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2


def run(image=""):
    # Defining kernels
    k_1 = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=np.float32) / 6
    k_2 = np.ones((3, 3), dtype=np.float32) / 9
    k_3 = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]], dtype=np.float32) / 10
    k_4 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16

    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_1 = cv2.filter2D(img, -1, k_1)
    r_2 = cv2.filter2D(img, -1, k_2)
    r_3 = cv2.filter2D(img, -1, k_3)
    r_4 = cv2.filter2D(img, -1, k_4)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(10, 10))
    axs[0, 0].set_title("Original")
    axs[0, 0].set_axis_off()
    axs[0, 0].imshow(img)
    axs[0, 1].set_title("Result 1")
    axs[0, 1].set_axis_off()
    axs[0, 1].imshow(r_1)
    axs[1, 0].set_title("Result 2")
    axs[1, 0].set_axis_off()
    axs[1, 0].imshow(r_2)
    axs[1, 1].set_title("Result 3")
    axs[1, 1].set_axis_off()
    axs[1, 1].imshow(r_3)
    axs[2, 0].set_title("Result 4")
    axs[2, 0].set_axis_off()
    axs[2, 0].imshow(r_4)
    plt.show()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="path to image")
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
