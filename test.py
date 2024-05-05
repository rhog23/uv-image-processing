"""
A very simple openCV demo to display a random array in dedicated window
With some text added
Window will reappear if closed
hit `escape` to quit or ctrl-c in the console window
"""

import cv2
import numpy as np
import time


def show_noise():
    frame = 0
    t0 = time.time()
    white = np.zeros([512, 512, 3], dtype=np.uint8)
    white.fill(255)  # or img[:] = 255

    # define some text properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    while True:
        frame += 1
        img = np.random.randn(512, 512)
        # add text to the image
        cv2.putText(
            img,
            "time: {:0.3f}, frame #: {}".format(time.time() - t0, frame),
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType,
        )

        cv2.putText(
            white,
            "time: {:0.3f}, frame #: {}".format(time.time() - t0, frame),
            (5, 100),
            font,
            fontScale,
            (0, 0, 0),
            lineType,
        )

        # display the image
        cv2.imshow("random pixels!", img)
        cv2.imshow("white", white)

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_noise()
