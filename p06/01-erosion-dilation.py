import cv2, sys
import numpy as np
import argparse

max_elem = 2
max_kernel_size = 10
title_trackbar_element_shape = "Element"
title_trackbar_kernel_size = "Kernel size"
title_erosion_window = "Erosion Demo"
title_dilation_window = "Dilation Demo"


def main(image):
    global src
    src = cv2.imread(image)
    if src is None:
        print("File doesn't exists: ", image)
        sys.exit(0)

    cv2.namedWindow(title_erosion_window)
    cv2.createTrackbar(
        title_trackbar_element_shape, title_erosion_window, 0, max_elem, erosion
    )
    cv2.createTrackbar(
        title_trackbar_kernel_size, title_erosion_window, 0, max_kernel_size, erosion
    )

    cv2.namedWindow(title_dilation_window)
    cv2.createTrackbar(
        title_trackbar_element_shape, title_dilation_window, 0, max_elem, dilation
    )
    cv2.createTrackbar(
        title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, dilation
    )

    erosion(0)
    dilation(0)
    cv2.waitKey()


def morph_shape(val):
    shapes = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]

    return shapes[val]


def erosion(val):
    erosion_size = (
        2 * cv2.getTrackbarPos(title_trackbar_kernel_size, title_erosion_window) + 1
    )
    erosion_shape = morph_shape(
        cv2.getTrackbarPos(title_trackbar_element_shape, title_erosion_window)
    )

    element = cv2.getStructuringElement(erosion_shape, (erosion_size, erosion_size))

    result = cv2.erode(src, element)
    cv2.imshow(title_erosion_window, result)


def dilation(val):
    dilation_size = cv2.getTrackbarPos(
        title_trackbar_kernel_size, title_dilation_window
    )
    dilation_shape = morph_shape(
        cv2.getTrackbarPos(title_trackbar_element_shape, title_dilation_window)
    )

    element = cv2.getStructuringElement(
        dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1)
    )

    result = cv2.dilate(src, element)
    cv2.imshow(title_dilation_window, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to image")
    args = parser.parse_args()

    main(args.image)
