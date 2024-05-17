import numpy as np
import tensorflow as tf
import cv2, time, dlib, joblib
from centerface import CenterFace


# Face Detection
def detect_faces(img):
    tinggi, lebar = img.shape[:2]

    dets, _ = centerface(img, tinggi, lebar, threshold=0.25)

    if len(dets) == 1:
        for det in dets:
            bbox = det[:4]
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])

            cropped_face = (
                tf.image.crop_to_bounding_box(
                    img,
                    y_min,
                    x_min,
                    y_max - y_min,
                    x_max - x_min,
                )
                .numpy()
                .astype(np.uint8)
            )

            # resize the cropped face into 150 x 150 (as required by dlib's model);
            # adds padding to prevent changing the structure of the face
            resized_img = (
                tf.image.resize_with_pad(cropped_face[..., ::-1], 150, 150)
                .numpy()
                .astype(np.uint8)
            )

            # generate 128d embeddings using dlib's model
            face_desc = np.array(facerec.compute_face_descriptor(resized_img)).reshape(
                1, -1
            )
            print(len(face_desc))

            result = knn_model.predict(face_desc)
            print(result)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
            # return img[y:h, x:w]  # cropping the detected face

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


target_width = 320
target_height = 180
roi_width = 100
roi_height = 100

# instantiate CenterFace
centerface = CenterFace(landmarks=True)

# load dlib model
facerec = dlib.face_recognition_model_v1(
    "./models/dlib_face_recognition_resnet_model_v1.dat"
)

# load knn model
knn_model = joblib.load("svm_classifier.joblib")

top_left_x, top_left_y, bottom_right_x, bottom_right_y = calculate_roi(
    target_width, target_height, roi_width, roi_height
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)  # resizing the camera's width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)  # resizing the camera's height

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()

    # draw roi box
    cv2.rectangle(
        frame,
        (top_left_x, top_left_y),
        (bottom_right_x, bottom_right_y),
        (255, 255, 0),
        2,
    )

    rect_img = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = detect_faces(rect_img)

    new_frame_time = time.time()
    fps = f"[FPS]: {str(int(1 / (new_frame_time - prev_frame_time)))}"
    prev_frame_time = new_frame_time

    cv2.rectangle(frame, (0, 0), (80, 20), (0, 0, 0), -1)

    cv2.putText(
        frame,
        fps,
        (0, 15),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
    cv2.imshow("face detection demo", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
