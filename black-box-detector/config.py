import cv2
import numpy as np
import utils

win_name = "Black Box Configuration"

cv2.namedWindow(win_name)
cap = cv2.VideoCapture(0)


def gray_bgr(frame):
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def _empty(a):
    pass


cv2.createTrackbar("Gray Min", win_name, 0, 255, _empty)
cv2.createTrackbar("Gray Max", win_name, 0, 255, _empty)
cv2.createTrackbar("Blurring Kernel", win_name, 1, 21, _empty)

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (180, 100))

    gray_f = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eq_f = cv2.equalizeHist(gray_f)

    gray_min = cv2.getTrackbarPos("Gray Min", win_name)
    gray_max = cv2.getTrackbarPos("Gray Max", win_name)
    blur_kernel = cv2.getTrackbarPos("Blurring Kernel", win_name)

    # Blurring the frame
    blurred_f = utils.applyBlurring(eq_f, tuple([blur_kernel, blur_kernel]))

    mask = cv2.inRange(blurred_f, gray_min, gray_max)

    result_frames = np.concatenate(
        (frame, gray_bgr(gray_f), gray_bgr(eq_f), gray_bgr(blurred_f), gray_bgr(mask)),
        axis=1,
    )

    cv2.imshow(win_name, result_frames)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
