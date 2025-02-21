import cv2, time, utils
import numpy as np
from skimage import measure

video_capture = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0


def detect_black_box(image):
    image = cv2.convertScaleAbs(image, alpha=2.5, beta=0)
    text = ""
    image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_grayscale = cv2.equalizeHist(image_grayscale)

    # Blurring the frame
    blurred_f = utils.applyBlurring(image_grayscale, (7, 7))

    mask = cv2.inRange(blurred_f, 0, 40)

    label_image = measure.label(mask)
    for region in measure.regionprops(label_image):
        if region.area >= 10000 and region.area <= 90000:
            if region.area >= 10000 and region.area <= 51000:
                text = "small"
            elif region.area <= 60000:
                text = "medium"
            elif region.area <= 80000:
                text = "big"
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(image, (minc, minr), (maxc, maxr), (255, 134, 58), 1)
            cv2.putText(
                image,
                f"size: {text} | area: {region.area}",
                (minc, minr),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    # return image
    return np.expand_dims(mask, axis=-1)


while True:
    ret, image_frame = video_capture.read()
    height, width = image_frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # calculating the roi-box coordinates
    top_left_x = center_x - 150
    top_left_y = center_y - 150
    bottom_right_x = top_left_x + 300
    bottom_right_y = top_left_y + 300

    # suppress coordinates to stay within the image_frame bounds
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(width, bottom_right_x)
    bottom_right_y = min(height, bottom_right_y)

    image_frame = cv2.convertScaleAbs(image_frame, alpha=0.25, beta=0)

    cv2.rectangle(
        image_frame,
        (top_left_x, top_left_y),
        (bottom_right_x, bottom_right_y),
        (70, 57, 230),
        2,
    )
    rect_img = image_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    image_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = (
        detect_black_box(rect_img)
    )

    new_frame_time = time.time()
    fps = f"[FPS]: {str(int(1 / (new_frame_time - prev_frame_time)))}"
    prev_frame_time = new_frame_time

    cv2.putText(
        image_frame,
        fps,
        (0, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("black box size sorter", image_frame)

    if cv2.waitKey(1) == ord("q"):
        print("[INFO] Closing...")
        break

video_capture.release()
cv2.destroyAllWindows()
