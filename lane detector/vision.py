import cv2
import numpy as np


def preprocessFrame(frame: cv2.typing.MatLike) -> cv2.typing.MatLike:

    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    lower_white = np.array([0, 0, 210])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(hls, lower_white, upper_white)
    hls_result = cv2.bitwise_and(frame, frame, mask=mask)

    return hls_result
