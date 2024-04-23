import cv2
import typing


def applyBlurring(
    frame: cv2.typing.MatLike, kernel_size: typing.Tuple[int, ...]
) -> cv2.typing.MatLike:
    try:
        result = cv2.GaussianBlur(frame, kernel_size, 0)
    except cv2.error:
        result = frame
    return result


def edgeDet(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:
    result = cv2.Canny(frame, 100, 150)
    return result
