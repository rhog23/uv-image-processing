import cv2, sys
import argparse

max_elem = 2
max_kernel_size = 10
title_trackbar_element_shape = "Element shape"
title_trackbar_kernel_size = "Kernel size"
title_opening_window = "Opening Demo"
title_closing_window = "Closing Demo"

# img = cv2.imread("images/charizard.png", cv2.IMREAD_GRAYSCALE)
# img = cv2.GaussianBlur(img, (3, 3), 0)
# (thresh, img_bw) = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
# cv2.imshow("test", img_bw)
# cv2.waitKey(0)


def main(image):
    global src
    src = cv2.imread(image)
    if src is None:
        print("File doesn't exists: ", image)
        sys.exit()

    # Trackbar Element Shape
    # Opening
    cv2.namedWindow(title_opening_window)
    cv2.createTrackbar(
        title_trackbar_element_shape, title_opening_window, 0, max_elem, opening
    )

    # Closing
    cv2.namedWindow(title_closing_window)
    cv2.createTrackbar(
        title_trackbar_element_shape, title_closing_window, 0, max_elem, closing
    )

    # Trackbar Kernel Size
    # Opening
    cv2.namedWindow(title_opening_window)
    cv2.createTrackbar(
        title_trackbar_kernel_size, title_opening_window, 0, max_kernel_size, opening
    )

    # Closing
    cv2.namedWindow(title_closing_window)
    cv2.createTrackbar(
        title_trackbar_kernel_size, title_closing_window, 0, max_kernel_size, closing
    )

    opening(0)
    closing(0)
    cv2.waitKey()


def structuring_element(val):
    shapes = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]

    return shapes[val]


def opening(val):
    # kernel size always transformed into odd number
    kernel_size = (
        2 * cv2.getTrackbarPos(title_trackbar_kernel_size, title_opening_window) + 1
    )
    element_shape = structuring_element(
        cv2.getTrackbarPos(title_trackbar_element_shape, title_opening_window)
    )

    element = cv2.getStructuringElement(element_shape, (kernel_size, kernel_size))

    result = cv2.morphologyEx(src, cv2.MORPH_OPEN, element)
    cv2.imshow(title_opening_window, result)


def closing(val):
    # kernel size always transformed into odd number
    kernel_size = (
        2 * cv2.getTrackbarPos(title_trackbar_kernel_size, title_closing_window) + 1
    )
    element_shape = structuring_element(
        cv2.getTrackbarPos(title_trackbar_element_shape, title_closing_window)
    )

    element = cv2.getStructuringElement(element_shape, (kernel_size, kernel_size))

    result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, element)
    cv2.imshow(title_closing_window, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to image")
    args = parser.parse_args()
    main(args.image)
