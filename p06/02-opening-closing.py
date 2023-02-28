import cv2, sys
import argparse

src = None
max_elem = 2
max_kernel_size = 10
title_trackbar_element_shape = "Element shape"
title_trackbar_kernel_size = "Kernel size"
title_opening_window = "Opening Demo"
title_closing_window = "Closing Demo"


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


def structuring_element(shape, size):
    shapes = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]

    return cv2.getStructuringElement(shapes[shape], (size, size))


def opening(val):
    element = structuring_element(
        cv2.getTrackbarPos(title_trackbar_element_shape, title_opening_window),
        (2 * cv2.getTrackbarPos(title_trackbar_kernel_size, title_opening_window) + 1),
    )

    result = cv2.morphologyEx(src, cv2.MORPH_OPEN, element)
    cv2.imshow(title_opening_window, result)


def closing(val):
    element = structuring_element(
        cv2.getTrackbarPos(title_trackbar_element_shape, title_closing_window),
        (2 * cv2.getTrackbarPos(title_trackbar_kernel_size, title_closing_window) + 1),
    )

    result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, element)
    cv2.imshow(title_closing_window, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, required=True, help="path to image")
    args = parser.parse_args()
    main(args.image)
