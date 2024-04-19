#!/usr/bin/env python3

import cv2
from vision import preprocessFrame

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = preprocessFrame(frame)
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break
cap.release()
cv2.destroyAllWindows()
