import cv2
import sys, time
import numpy as np
from skimage import measure

video_capture = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0

aoi_top_left = (200, 75)
aoi_bottom_right = (520, 425)


def detect_black_box(image):
    image = cv2.convertScaleAbs(image, alpha=2.5, beta=0)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blurred = cv2.GaussianBlur(image_grayscale, (7, 7), 0)
    (T, threshInv) = cv2.threshold(image_blurred, 50, 255, cv2.THRESH_BINARY_INV)
    label_image = measure.label(threshInv)
    for region in measure.regionprops(label_image):
        if region.area >= 26800 and region.area <= 90000:
            if region.area >= 26800 and region.area <= 51000:
                text = "small"
            elif region.area <= 60000:
                text = "medium"
            elif region.area <= 80000:
                text = "big"
            print(region.area)
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(image, (minc, minr), (maxc, maxr), (255, 134, 58), 1)
            cv2.putText(
                image,
                text,
                (minc, minr),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return image


while True:
    ret, image_frame = video_capture.read()
    image_frame = cv2.convertScaleAbs(image_frame, alpha=0.25, beta=0)

    cv2.rectangle(image_frame, aoi_top_left, aoi_bottom_right, (70, 57, 230), 2)
    rect_img = image_frame[
        aoi_top_left[1] : aoi_bottom_right[1], aoi_top_left[0] : aoi_bottom_right[0]
    ]

    image_frame[
        aoi_top_left[1] : aoi_bottom_right[1], aoi_top_left[0] : aoi_bottom_right[0]
    ] = detect_black_box(rect_img)

    new_frame_time = time.time()
    fps = f"[FPS]: {str(int(1 / (new_frame_time - prev_frame_time)))}"
    prev_frame_time = new_frame_time

    cv2.putText(
        image_frame,
        fps,
        (0, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (236, 56, 131),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("black box size sorter", image_frame)

    if cv2.waitKey(1) == ord("q"):
        print("[INFO] Closing...")
        break

video_capture.release()
cv2.destroyAllWindows()
