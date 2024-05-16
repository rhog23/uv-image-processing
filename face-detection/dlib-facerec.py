import sys, os, dlib, logging, cv2, keras
from centerface import CenterFace
import tensorflow as tf
import numpy as np

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


def test_centerface(img):
    height, width = img.shape[:2]
    faces, lms = face_detector(img, height, width, threshold=0.35)

    for det in faces:
        boxes, score = det[:4], det[4]
        cv2.rectangle(
            img,
            (int(boxes[0]), int(boxes[1])),
            (int(boxes[2]), int(boxes[3])),
            (2, 255, 0),
            1,
        )

    for lm in lms:
        lm = np.array(lm).reshape(len(lm) // 2, 2)
        for point in lm:
            x, y = point
            cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), -1)

    return img


def test_dlib(img, detector, landmark_detector):
    cropped_image = []

    # detecting faces
    dets = detector(img, 1)

    logging.info(f"Number of detected faces: {len(dets)}")

    for k, d in enumerate(dets):
        # get the landmarks
        shape = landmark_detector(img, d)

        cv2.circle(img, (d.left(), d.top()), 2, (255, 255, 0), -1)
        cv2.circle(img, (d.right(), d.bottom()), 2, (255, 255, 0), -1)
        cv2.rectangle(
            img,
            (d.left(), d.top()),
            (d.right(), d.bottom()),
            (255, 0, 0),
            1,
            cv2.FONT_HERSHEY_SIMPLEX,
        )

        for part in shape.parts():
            cv2.circle(img, (part.x, part.y), 2, (0, 0, 255), 2)

    return cropped_image


if __name__ == "__main__":

    img_source = load_image(
        "test-image-2.jpg", target_width, target_height, True
    )  # RGB Image. Preserved ratio when resizing the image

    centerface_result = test_centerface(img_source.copy())

    result = np.hstack([img_source[..., ::-1]])

    cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
    cropped_image = (
        tf.image.crop_to_bounding_box(
            rgb_image, d.top(), d.left(), d.height(), d.width()
        )
        .numpy()
        .astype(np.uint8)
    )


    face_chip = cv2.cvtColor(dlib.get_face_chip(rgb_image, shape), cv2.COLOR_RGB2BGR)

    face_chip_150 = (
        tf.image.resize_with_pad(dlib.get_face_chip(cropped_image, shape), 150, 150)
        .numpy()
        .astype(np.uint8)
    )

    original_face_dec = np.array(
        list(
            facerec.compute_face_descriptor(
                tf.image.resize(cropped_image, (150, 150)).numpy().astype(np.uint8)
            )
        )
    )

    aligned_face_dec = np.array(list(facerec.compute_face_descriptor(face_chip_150)))

    # euc_distance = np.linalg.norm(aligned_face_dec - original_face_dec)
    euc_distance = np.sqrt(np.sum(np.square(aligned_face_dec - original_face_dec)))
    print(euc_distance)
"""
