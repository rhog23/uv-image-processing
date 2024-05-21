import numpy as np

# import pandas as pd
import tensorflow as tf
from pathlib import Path
from backbones import get_model
from centerface import CenterFace
import logging, keras, os, cv2, dlib, torch, pickle

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


def preprocess_face(img):
    resized = tf.image.resize_with_pad(img, 112, 112).numpy().astype(np.uint8)
    transposed = np.transpose(resized, (2, 0, 1))
    result = torch.from_numpy(transposed).unsqueeze(0).float()
    result.div_(255).sub_(0.5).div_(0.5)

    return result


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


@torch.no_grad()
def inference(img, net):
    feat = net(img)
    return feat.numpy()


# instantiate CenterFace
face_detector = CenterFace()

# load dlib's face recognition model
# facerec = dlib.face_recognition_model_v1(
#     "./models/dlib_face_recognition_resnet_model_v1.dat"
# )

net = get_model("edgeface_xs_gamma_06", fp16=False)
net.load_state_dict(
    torch.load("checkpoints/edgeface_xs_gamma_06.pt", map_location="cpu")
)
net.eval()

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
    # resized_img = tf.image.resize_with_pad(result, 150, 150).numpy().astype(np.uint8)
    resized_img = preprocess_face(result)

    # generate 128d embeddings using dlib's model
    # face_desc = np.array(facerec.compute_face_descriptor(resized_img)).tolist()

    # generate 512d embbeddings using edgeface's model
    face_desc = inference(resized_img, net)

    list_embeddings.append(face_desc[0])

# dictionary of names and the associated facial embeddings
data = {"name": list_names, "embedding": list_embeddings}

f = open("embeddings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()

# create dataframe and saving to .csv
# df = pd.DataFrame(data)

# saving dataframe
# df.to_csv("facial-embeddings-v2.csv")

# sanity check
# print(df.head())
