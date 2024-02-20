import cv2
import time
import numpy as np
from skimage import measure, filters, color, morphology, util

video_capture = cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
prev_frame_time = 0
new_frame_time = 0


def detect_black_box(image):
    image = cv2.convertScaleAbs(image, alpha=2.5, beta=0)
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image_blurred = filters.gaussian(image_grayscale, sigma=2)
    thresh = filters.threshold_otsu(util.img_as_float(image_grayscale))
    mask = image_grayscale > thresh
    label_image = measure.label(mask)
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

    return np.expand_dims(thresh, axis=-1)


while True:
    ret, image_frame = video_capture.read()
    height, width = image_frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # calculating the roi-box coordinates
    top_left_x = center_x - 240
    top_left_y = center_y - 75
    bottom_right_x = top_left_x + 480
    bottom_right_y = top_left_y + 150

    # suppress coordinates to stay within the image_frame bounds
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(width, bottom_right_x)
    bottom_right_y = min(height, bottom_right_y)

    image_frame = cv2.convertScaleAbs(image_frame, alpha=0.25, beta=0)

    cv2.rectangle(image_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (70, 57, 230), 2)
    rect_img = image_frame[
        top_left_y : bottom_right_y, top_left_x : bottom_right_x
    ]

    image_frame[
        top_left_y : bottom_right_y, top_left_x : bottom_right_x
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
