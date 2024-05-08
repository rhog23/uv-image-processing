import cv2, time
import numpy as np
from centerface import CenterFace


# Face Detection
def detect_faces(img):
    img = cv2.convertScaleAbs(img, alpha=2.5, beta=0)
    tinggi, lebar = img.shape[:2]

    dets, _ = centerface(img, tinggi, lebar, threshold=0.25)

    for det in dets:
        bbox = det[:4]
        y = int(bbox[1])
        h = int(bbox[3])
        x = int(bbox[0])
        w = int(bbox[2])
        cv2.rectangle(img, (x, y), (w, h), (255, 255, 255), 2)
        # return img[y:h, x:w] # cropping the detected face

    return img


def calculate_roi(width, height, roi_width, roi_height):
    center_x = width // 2
    center_y = height // 2

    # calculating the roi-box coordinates
    top_left_x = center_x - (roi_width // 2)
    top_left_y = center_y - (roi_height // 2)
    bottom_right_x = center_x + (roi_width // 2)
    bottom_right_y = center_y + (roi_height // 2)

    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(width, bottom_right_x)
    bottom_right_y = min(height, bottom_right_y)

    return (top_left_x, top_left_y, bottom_right_x, bottom_right_y)


def get_contour(image):
    """
    On Progress
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(
        gray_image, cv2.CV_64F, 1, 0, ksize=3
    )  # Sobel for horizontal edges
    sobely = cv2.Sobel(
        gray_image, cv2.CV_64F, 0, 1, ksize=3
    )  # Sobel for vertical edges

    sobel = np.absolute(sobelx) + np.absolute(sobely)
    thresh = 0.3 * np.max(sobel)  # Adjust threshold as needed
    binary_edges = np.where(sobel > thresh, 255, 0).astype(
        np.uint8
    )  # Convert to uint8 for display

    edges_image = cv2.cvtColor(
        binary_edges, cv2.COLOR_GRAY2BGR
    )  # Convert binary edges back to BGR
    cv2.addWeighted(image, 0.7, edges_image, 0.3, 0)  # Overlay with transparency


target_width = 320
target_height = 240
roi_width = 120
roi_height = 120
centerface = CenterFace(landmarks=True)
top_left_x, top_left_y, bottom_right_x, bottom_right_y = calculate_roi(
    target_width, target_height, roi_width, roi_height
)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)  # resizing the camera's width
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)  # resizing the camera's height

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()

    frame = cv2.convertScaleAbs(frame, alpha=0.25, beta=0)

    # draw roi box
    cv2.rectangle(
        frame,
        (top_left_x, top_left_y),
        (bottom_right_x, bottom_right_y),
        (255, 255, 0),
        2,
    )

    rect_img = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

    # frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = detect_faces(rect_img)
    result = detect_faces(rect_img)

    new_frame_time = time.time()
    fps = f"[FPS]: {str(int(1 / (new_frame_time - prev_frame_time)))}"
    prev_frame_time = new_frame_time

    cv2.putText(
        frame,
        fps,
        (0, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (236, 56, 131),
        1,
        cv2.LINE_AA,
    )

    cv2.imshow("face detection demo", result)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
