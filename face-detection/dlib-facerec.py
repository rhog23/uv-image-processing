import sys, os, dlib, logging, cv2, keras
from centerface import CenterFace
import tensorflow as tf
import numpy as np

logging.basicConfig(level=logging.INFO)


def load_image(path, target_width=0, target_height=0, preserve_ratio=False):
    # load image using keras
    img_arr = keras.utils.img_to_array(keras.utils.load_img(path))

    if target_width and target_height:
        # resize image to target width and height
        img_arr = tf.image.resize(
            img_arr, (target_height, target_width), preserve_aspect_ratio=preserve_ratio
        )

    return img_arr.numpy().astype(
        np.uint8
    )  # always convert type as np.uint8 if we want to show the image using opencv


def test_centerface(img, detector):
    cropped_image = None
    height, width = img.shape[:2]
    faces, lms = detector(img, height, width, threshold=0.35)

    for det in faces:
        boxes, score = det[:4], det[4]
        x_min = int(boxes[0])
        y_min = int(boxes[1])
        x_max = int(boxes[2])
        y_max = int(boxes[3])

        cropped_image = (
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

        # draw bounding box
        cv2.rectangle(
            img,
            (x_min, y_min),
            (x_max, y_max),
            (2, 255, 0),
            1,
        )

    for lm in lms:
        lm = np.array(lm).reshape(len(lm) // 2, 2)
        for point in lm:
            x, y = point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    return img, cropped_image


def test_dlib(img, detector, landmark_detector):
    cropped_image = None
    landmarks = []

    # detecting faces
    dets = detector(img, 1)

    logging.info(f"[DLIB] Number of detected faces: {len(dets)}")

    for _, d in enumerate(dets):
        # get the landmarks
        shape = landmark_detector(img, d)
        landmarks.append(shape)

        cropped_image = (
            tf.image.crop_to_bounding_box(img, d.top(), d.left(), d.height(), d.width())
            .numpy()
            .astype(np.uint8)
        )

        # draw bounding box
        cv2.rectangle(
            img,
            (d.left(), d.top()),
            (d.right(), d.bottom()),
            (255, 0, 0),
            1,
            cv2.FONT_HERSHEY_SIMPLEX,
        )

        # draw landmarks
        for part in shape.parts():
            cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), 2)

    return img, cropped_image, landmarks


def resize_crop(img, target_width=150, target_height=150, add_padding=False):
    if add_padding:
        img = (
            tf.image.resize_with_pad(img, target_width, target_height)
            .numpy()
            .astype(np.uint8)
        )
    else:
        img = (
            tf.image.resize(img, (target_width, target_height)).numpy().astype(np.uint8)
        )

    return img


if __name__ == "__main__":

    target_height = 240
    target_width = 320

    # load dlib's  face detector
    detector = dlib.get_frontal_face_detector()

    # load dlib's facial landmarks predictor
    sp = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")

    # load dlib's face recognition model
    facerec = dlib.face_recognition_model_v1(
        "./models/dlib_face_recognition_resnet_model_v1.dat"
    )

    # instantiate centerface detector
    face_detector = CenterFace()

    img_source = load_image(
        "test-image-2.jpg", target_width, target_height, False
    )  # RGB Image. Preserved ratio when resizing the image

    centerface_result, centerface_crop = test_centerface(
        img_source.copy(), face_detector
    )
    dlib_result, dlib_crop, landmarks = test_dlib(img_source.copy(), detector, sp)

    face_chip = dlib.get_face_chip(img_source.copy(), landmarks[0], padding=0)

    dlib_desc = np.array(
        facerec.compute_face_descriptor(resize_crop(dlib_crop, add_padding=True))
    )
    centerface_desc = np.array(
        facerec.compute_face_descriptor(resize_crop(centerface_crop, add_padding=True))
    )

    distance = np.linalg.norm(centerface_desc - dlib_desc)

    logging.info(f"Distance: {distance}")

    result = np.hstack(
        [img_source[..., ::-1], centerface_result[..., ::-1], dlib_result[..., ::-1]]
    )

    crop_result = np.hstack(
        [
            resize_crop(centerface_crop[..., ::-1], add_padding=True),
            resize_crop(dlib_crop[..., ::-1], add_padding=True),
        ]
    )

    cv2.imshow("result", result)
    cv2.imshow("cropped result", crop_result)
    cv2.imshow("face chip", face_chip[..., ::-1])

    cv2.waitKey(0)
    cv2.destroyAllWindows()
