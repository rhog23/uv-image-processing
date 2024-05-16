import os, cv2, dlib
import numpy as np
import pandas as pd
import logging, keras
import tensorflow as tf
from pathlib import Path
from centerface import CenterFace

logging.basicConfig(level=logging.INFO)

target_height: int = 640
target_width: int = 640

folder = Path("./foto-kelompok")

image_paths = sorted(folder.rglob("*/*.jpg"))

list_embeddings = []
list_names = []
# print(os.path.split(image_paths[0])[0].split(os.path.sep))


def load_image(path, target_width=0, target_height=0, preserve_ratio=True):
    # load image using keras
    img_arr = keras.utils.img_to_array(keras.utils.load_img(path))

    if target_width or target_height:
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
    faces, _ = detector(img, height, width, threshold=0.35)

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

    return cropped_image


# instantiate CenterFace
face_detector = CenterFace()

# load dlib's face recognition model
facerec = dlib.face_recognition_model_v1(
    "./models/dlib_face_recognition_resnet_model_v1.dat"
)

for path in image_paths:
    img_source = load_image(path, target_width, target_height)  # loads image
    name = os.path.split(path)[0].split(os.path.sep)[
        -1
    ]  # extracting the person's name from current path

    list_names.append(name)  # append to the list of names

    # face detection
    result = detect_face(img_source, face_detector)

    # resize the cropped face into 150 x 150 (as required by dlib's model);
    # adds padding to prevent changing the structure of the face
    resized_img = tf.image.resize_with_pad(result, 150, 150).numpy().astype(np.uint8)

    # generate 128d embeddings using dlib's model
    face_desc = np.array(facerec.compute_face_descriptor(resized_img))

    list_embeddings.append(face_desc)

# dictionary of names and the associated facial embeddings
data = {"name": list_names, "embedding": list_embeddings}

# create dataframe and saving to .csv
df = pd.DataFrame(data)

# saving dataframe
df.to_csv("facial-embeddings.csv")

# sanity check
print(df.head())
