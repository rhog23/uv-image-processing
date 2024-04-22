import cv2, utils, vision
import numpy as np

cap = cv2.VideoCapture(0)

gray_win_name = "Gray Config"


def _empty(a):
    pass


curve_list = []
average_val = 10

cv2.namedWindow(gray_win_name)
cv2.createTrackbar("Gray Min", gray_win_name, 0, 255, _empty)
cv2.createTrackbar("Gray Max", gray_win_name, 0, 255, _empty)
cv2.createTrackbar("Blurring Kernel", gray_win_name, 1, 21, _empty)

cap = cv2.VideoCapture(0)
init_points = [102, 80, 20, 160]
utils.init_trackball(init_points, gray_win_name)

while True:
    _, frame = cap.read()

    frame = cv2.resize(frame, (240, 180))
    frame_result = frame.copy()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adding trackbars for real-time change
    gray_min = cv2.getTrackbarPos("Gray Min", gray_win_name)
    gray_max = cv2.getTrackbarPos("Gray Max", gray_win_name)

    blur_kernel = cv2.getTrackbarPos("Blurring Kernel", gray_win_name)

    # warping points trackball
    top_left = cv2.getTrackbarPos("Top LR", gray_win_name)
    top_right = cv2.getTrackbarPos("Top UD", gray_win_name)
    bottom_left = cv2.getTrackbarPos("Bottom LR", gray_win_name)
    bottom_right = cv2.getTrackbarPos("Bottom UD", gray_win_name)

    points = np.float32(
        [
            (top_left, top_right),
            (240 - top_left, top_right),
            (bottom_left, bottom_right),
            (240 - bottom_left, bottom_right),
        ]
    )

    blurred_frame = vision.applyBlurring(gray_frame, tuple([blur_kernel, blur_kernel]))

    mask = cv2.inRange(blurred_frame, gray_min, gray_max)

    original_warp_frame = utils.draw_wrap_points(blurred_frame.copy(), points)
    warp_frame = utils.wrap_frame(mask, points, 240, 180)

    middlepoint, frame_hist = utils.get_histogram(
        warp_frame, display=True, min_per=0.5, region=4
    )

    curve_avg_pts, frame_hist = utils.get_histogram(
        warp_frame, display=True, min_per=0.9
    )

    curve_raw = curve_avg_pts - middlepoint

    curve_list.append(curve_raw)

    if len(curve_list) > average_val:
        curve_list.pop(0)

    curve = int(sum(curve_list) / len(curve_list))

    frame_invert_warp = utils.wrap_frame(warp_frame, points, 240, 180, invert=True)
    frame_invert_warp = cv2.cvtColor(frame_invert_warp, cv2.COLOR_GRAY2BGR)
    frame_invert_warp[0 : 180 // 3, 0:240] = 0, 0, 0

    frame_lane_color = np.zeros_like(frame)
    frame_lane_color[:] = 0, 255, 0
    frame_lane_color = cv2.bitwise_and(frame_invert_warp, frame_lane_color)
    frame_result = cv2.addWeighted(frame_result, 1, frame_lane_color, 1, 0)
    midY = 110

    cv2.putText(
        frame_result,
        str(curve),
        (240 // 2 - 80, 85),
        cv2.FONT_HERSHEY_COMPLEX,
        2,
        (255, 0, 255),
        3,
    )

    cv2.line(
        frame_result, (240 // 2, midY), (240 // 2 + (curve * 3), midY), (255, 0, 255), 5
    )

    cv2.line(
        frame_result,
        ((240 // 2 + (curve * 3)), midY - 25),
        (240 // 2 + (curve * 3), midY + 25),
        (0, 255, 0),
        5,
    )

    for x in range(-30, 30):
        w = 240 // 20
        cv2.line(
            frame_result,
            (w * x + int(curve // 50), midY - 10),
            (w * x + int(curve // 50), midY + 10),
            (0, 0, 255),
            2,
        )

    horizontal_concat_0 = np.concatenate(
        (
            frame,
            cv2.cvtColor(blurred_frame, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        ),
        axis=1,
    )
    horizontal_concat_1 = np.concatenate(
        (
            cv2.cvtColor(original_warp_frame, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(warp_frame, cv2.COLOR_GRAY2BGR),
            frame_result,
        ),
        axis=1,
    )
    vertical_concat = np.concatenate((horizontal_concat_0, horizontal_concat_1), axis=0)
    cv2.imshow(gray_win_name, vertical_concat)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
