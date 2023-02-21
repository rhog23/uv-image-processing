import argparse
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt


def run(image=""):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ripple_code = {"0", "1", "2"}

    print(
        "Ripple Type:\n0: vertical ripple\n1: horizontal ripple\n2: vertical and horizontal ripple"
    )
    ripple_type = input("Enter Ripple Type: ")

    if ripple_type not in ripple_code:
        print("Invalid input")
        return False

    wave_height = int(input("Enter wave height: "))
    wave_length = int(input("Enter wave length: "))

    if ripple_type == "0":
        result = vertical_ripple(img, wave_height, wave_length)
    elif ripple_type == "1":
        result = horizontal_ripple(img, wave_height, wave_length)
    elif ripple_type == "2":
        result = ver_hor_ripple(img, wave_height, wave_length)

    show_result(img, result)


def vertical_ripple(img, wave_height=10, wave_length=180):
    result = np.zeros(img.shape, dtype=img.dtype)
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            offset_x = int(wave_height * math.sin(2 * math.pi * i / wave_length))
            offset_y = 0
            if j + offset_x < height:
                result[i, j] = img[i, (j + offset_x) % width]
            else:
                result[i, j] = 0

    return result


def horizontal_ripple(img, wave_height=10, wave_length=180):
    result = np.zeros(img.shape, dtype=img.dtype)
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            offset_x = 0
            offset_y = int(wave_height * math.sin(2 * math.pi * j / wave_length))
            if i + offset_y < height:
                result[i, j] = img[(i + offset_y) % height, j]
            else:
                result[i, j] = 0

    return result


def ver_hor_ripple(img, wave_height=10, wave_length=180):
    result = np.zeros(img.shape, dtype=img.dtype)
    height, width = img.shape[:2]
    for i in range(height):
        for j in range(width):
            offset_x = int(wave_height * math.sin(2 * math.pi * i / wave_length))
            offset_y = int(wave_height * math.sin(2 * math.pi * j / wave_length))
            if i + offset_y < height and j + offset_x < width:
                result[i, j] = img[(i + offset_y) % height, (j + offset_x) % width]
            else:
                result[i, j] = 0

    return result


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
    opt = parser.parse_args()
    print(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
