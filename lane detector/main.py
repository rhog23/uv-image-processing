#!/usr/bin/env python3

from distutils.command import clean
import cv2
import numpy as np


def preprocessing(frame):
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (7, 7))

    mask = cv2.inRange(blur, 0, 120)
    
    kernel = 

    return mask


def cleanup(cap):
    cap.release()
    cv2.destroyAllWindows()


cap = cv2.VideoCapture(0)
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
