import glob
import numpy as np
import logging, keras
import tensorflow as tf
from pathlib import Path
from centerface import CenterFace

logging.basicConfig(level=logging.INFO)

target_width: int = 320

images_path = ""

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


def detect_face(img, detector):
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

    return img, cropped_image


# instantiate CenterFace
face_detector = CenterFace()
