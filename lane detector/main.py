#!/usr/bin/env python3

import cv2
import numpy as np


def preprocessing(frame):
    height, width = frame.shape[:2]

    frame = cv2.rotate(frame, cv2.ROTATE_180)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))

    mask = cv2.inRange(blur, 0, 120)

    erode_kernel = np.ones(
        (
            5,
            5,
        ),
        np.uint8,
    )
    dilate_kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, erode_kernel, iterations=1)
    dilated_mask = cv2.dilate(eroded_mask, dilate_kernel, iterations=1)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    results = {}

    for cnt in contours:
        M = cv2.moments(contours[0])

        if M["m00"] == 0:
            cx, cy = width, 0
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        cv2.circle(gray, (cx, cy), 5, (255, 255, 255), -1)
        results["centroid"] = (cx, cy)

    return np.hstack([gray, blur, mask, eroded_mask, dilated_mask])


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 160)
cap.set(4, 120)

while True:
    try:
        ret, frame = cap.read()

        result = preprocessing(frame)

        cv2.imshow("frame", result)

        if cv2.waitKey(1) == ord("q"):
            cleanup(cap)
            break
    except KeyboardInterrupt:
        cleanup(cap)
        exit()
