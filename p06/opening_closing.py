import argparse
import functools
import cv2

max_elem: int = 2
max_kernel_size: int = 10
title_trackbar_element_shape: str = "Element shape"
title_trackbar_kernel_size: str = "Kernel size"
title_opening_window: str = "Opening Demo"
title_closing_window: str = "Closing Demo"


def main(image):
    try:
        src = cv2.imread(image)

        # opening window
        # structuring element shape
        cv2.namedWindow(title_opening_window)
        cv2.createTrackbar(
            title_trackbar_element_shape,
            title_opening_window,
            0,
            max_elem,
            functools.partial(opening, src=src),
        )

        # structuring element size
        cv2.createTrackbar(
            title_trackbar_kernel_size,
            title_opening_window,
            0,
            max_kernel_size,
            functools.partial(opening, src=src),
        )

        # closing window
        # structuring element shape
        cv2.namedWindow(title_closing_window)
        cv2.createTrackbar(
            title_trackbar_element_shape,
            title_closing_window,
            0,
            max_elem,
            functools.partial(closing, src=src),
        )

        # structuring element size
        cv2.namedWindow(title_closing_window)
        cv2.createTrackbar(
            title_trackbar_kernel_size,
            title_closing_window,
            0,
            max_kernel_size,
            functools.partial(closing, src=src),
        )

        opening(0, src)
        closing(0, src)
        cv2.waitKey()

    except Exception as e:
        print(f"!!!Terjadi kesalahan!!!\nINFO: {e}")


def structuring_element(shape: int, size: int) -> cv2.typing.MatLike:
    shapes = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]

    return cv2.getStructuringElement(shapes[shape], (size, size))


def opening(val: int, src: cv2.typing.MatLike) -> None:
    """Fungsi yang digunakan untuk menerapkan opening (erosi -> dilasi) pada citra

    Args:
        val (int): parameter default yang dibutuhkan oleh OpenCV untuk mengambil value trackbar. Tidak dipanggil di dalam fungsi
        src (cv2.typing.MatLike): citra
    """
    se_type = cv2.getTrackbarPos(title_trackbar_element_shape, title_opening_window)
    se_size = (
        2 * cv2.getTrackbarPos(title_trackbar_kernel_size, title_opening_window) + 1
    )
    element = structuring_element(
        se_type,
        se_size,
    )

    result = cv2.morphologyEx(src, cv2.MORPH_OPEN, element)
    cv2.imshow(title_opening_window, result)


def closing(val: int, src: cv2.typing.MatLike) -> None:
    """Fungsi yang digunakan untuk menerapkan closing (dilasi -> erosi) pada citra

    Args:
        val (int): parameter default yang dibutuhkan oleh OpenCV untuk mengambil value trackbar. Tidak dipanggil di dalam fungsi
        src (cv2.typing.MatLike): citra
    """
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
