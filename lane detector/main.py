#!/usr/bin/env python3

import cv2
import vision
import numpy as np

DIRECTION = 0

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 320)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            gray = vision.applyBlurring(gray, tuple([9, 9]))

            ret, th = cv2.threshold(gray, 0, 55, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            edges = vision.edgeDet(th)
            edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8))

            masked = cv2.bitwise_and(th, th, mask=edges)

            contours, _ = cv2.findContours(
                masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            marked_frame = frame.copy()
            cv2.drawContours(marked_frame, contours, -1, (255, 255, 0), 2)
            cv2.imshow("frame", marked_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
cap.release()
cv2.destroyAllWindows()
