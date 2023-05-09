import time
import cv2 as cv
import sys
import numpy as np

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 160)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 160)
cap.set(cv.CAP_PROP_FPS, 36)

if not cap.isOpened():
    print("[INFO] Unable to open camera ...")
    sys.exit()

previous_frame_time = 0
new_frame_time = 0

low_black = np.array([0, 0, 121])
high_black = np.array([255, 255, 255])

while True:
    ret, frame = cap.read()

    if not ret:
        print("[INFO] Can't receive frame (stream end?). Exiting ...")
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, low_black, high_black)
    frame = cv.bitwise_and(frame, frame, mask=mask)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - previous_frame_time)
    previous_frame_time = new_frame_time

    fps = str(int(fps))

    cv.putText(
        frame,
        fps,
        (0, 50),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        3,
        cv.LINE_AA,
    )

    cv.imshow("frame", frame)
    if cv.waitKey(1) == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
