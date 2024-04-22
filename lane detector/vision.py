import cv2, typing
import numpy as np
import utils


def filterColor(
    frame: cv2.typing.MatLike,
    color_code: typing.AnyStr,
    lower: typing.List[typing.Union[int, int, int]],
    upper: typing.List[typing.Union[int, int, int]],
) -> cv2.typing.MatLike:

    converted_frame = cv2.cvtColor(frame, utils.COLOR_CHANNELS_CONVERSION[color_code])
    lower_bound = np.array(lower)
    upper_bound = np.array(upper)
    mask = cv2.inRange(converted_frame, lower_bound, upper_bound)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    return result


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
