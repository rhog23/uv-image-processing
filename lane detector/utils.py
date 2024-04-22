import typing
import cv2
import numpy as np

COLOR_CHANNELS_CONVERSION: typing.Dict = {
    "gray": cv2.COLOR_BGR2GRAY,
    "rgb": cv2.COLOR_BGR2RGB,
    "hsv": cv2.COLOR_BGR2HSV,
    "hsl": cv2.COLOR_BGR2HLS,
}


def _empty_trackball_callback(_null):
    pass


def init_trackball(init_val, win_name, target_width=240, target_height=180):
    cv2.namedWindow(win_name)
    # cv2.resizeWindow(win_name, 360, 240)
    cv2.createTrackbar(
        "Top LR", win_name, init_val[0], target_width, _empty_trackball_callback
    )

    cv2.createTrackbar(
        "Top UD",
        win_name,
        init_val[1],
        target_height,
        _empty_trackball_callback,
    )

    cv2.createTrackbar(
        "Bottom LR",
        win_name,
        init_val[2],
        target_width,
        _empty_trackball_callback,
    )

    cv2.createTrackbar(
        "Bottom UD",
        win_name,
        init_val[3],
        target_height,
        _empty_trackball_callback,
    )


def draw_wrap_points(frame, points):
    for x in range(4):
        cv2.circle(
            frame,
            (int(points[x][0]), int(points[x][1])),
            5,
            (0, 0, 0),
            cv2.FILLED,
        )

    return frame


def get_histogram(frame, min_per=0.1, display=False, region=1):
    if region == 1:
        hist_values = np.sum(frame, axis=0)
    else:
        hist_values = np.sum(frame[frame.shape[0] // region :, :], axis=0)

    max_value = np.max(hist_values)
    min_value = min_per * max_value

    idx_array = np.where(hist_values >= min_value)
    base_point = int(np.average(idx_array))

    if display:
        frame_hist = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

        for x, intensity in enumerate(hist_values):
            cv2.line(
                frame_hist,
                (x, frame.shape[0]),
                (x, frame.shape[0] - intensity // 255 // region),
                (255, 0, 255),
                1,
            )

            cv2.circle(
                frame_hist, (base_point, frame.shape[0]), 20, (0, 255, 255), cv2.FILLED
            )
        return base_point, frame_hist

    return base_point


def wrap_frame(frame, points, width, height, invert=False):
    source_points = np.float32(points)
    target_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    if invert:
        # swap source points and target points
        matrix = cv2.getPerspectiveTransform(target_points, source_points)
    else:
        matrix = cv2.getPerspectiveTransform(source_points, target_points)

    frame_wrap = cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
    )

    return frame_wrap
