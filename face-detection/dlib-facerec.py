import sys, os, dlib, glob, cv2
from centerface import CenterFace
import tensorflow as tf
import numpy as np

target_height = 240
target_width = 320


# load dlib face detector
detector = dlib.get_frontal_face_detector()

# load facial landmarks predictor
sp = dlib.shape_predictor("./models/shape_predictor_5_face_landmarks.dat")

# load face recognition model
facerec = dlib.face_recognition_model_v1(
    "./models/dlib_face_recognition_resnet_model_v1.dat"
)

image = cv2.imread("test-image-2.jpg")
image = cv2.resize(image, (target_width, target_height))
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

dets = detector(rgb_image, 1)
print(f"Number of detected faces: {len(dets)}")

for k, d in enumerate(dets):
    print(
        f"Detection {d}: Left: {d.left()}, Top: {d.top()}, Right: {d.right()}, Bottom: {d.bottom()}"
    )

    cropped_image = (
        tf.image.crop_to_bounding_box(
            rgb_image, d.top(), d.left(), d.height(), d.width()
        )
        .numpy()
        .astype(np.uint8)
    )

    cv2.circle(image, (d.left(), d.top()), 2, (255, 255, 0), 2)
    cv2.circle(image, (d.right(), d.bottom()), 2, (255, 255, 0), 2)
    cv2.rectangle(
        image,
        (d.left(), d.top()),
        (d.right(), d.bottom()),
        (255, 0, 0),
        1,
        cv2.FONT_HERSHEY_SIMPLEX,
    )

    # predict landmarks
    shape = sp(rgb_image, d)

    print("Landmarks", shape.parts())
    for part in shape.parts():

        cv2.circle(rgb_image, (part.x, part.y), 2, (0, 0, 255), 2)

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

    # resized into 320, 240
    face_chip = (
        tf.image.resize_with_pad(face_chip, target_height, target_width)
        .numpy()
        .astype(np.uint8)
    )

    cropped_image = (
        tf.image.resize_with_pad(cropped_image, target_height, target_width)
        .numpy()
        .astype(np.uint8)
    )

    # result = np.hstack(
    #     [
    #         face_chip_150,
    #         tf.image.resize_with_pad(cropped_image, 150, 150)
    #         .numpy()
    #         .astype(np.uint8)[..., ::-1],
    #     ]
    # )

cv2.imshow("result", rgb_image[..., ::-1])

cv2.waitKey(0)
cv2.destroyAllWindows()
