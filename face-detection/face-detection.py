import numpy as np
import tensorflow as tf
from backbones import get_model
from centerface import CenterFace
import cv2, time, joblib, torch, face_recognition, logging, pickle

logging.basicConfig(level=logging.INFO)
data = pickle.loads(open("embeddings.pickle", "rb").read())
names = ["Alvandy", "Gres", "Jacky", "Kelvin", "Yonathan"]


@torch.no_grad()
def inference(img, net):
    feat = net(img)
    return feat.numpy()


# Face Detection
def detect_face(img):
    tinggi, lebar = img.shape[:2]

    x_min, y_min, x_max, y_max = 0, 0, 0, 0

    dets, _ = centerface(img, tinggi, lebar, threshold=0.25)

    if len(dets) == 1:
        for det in dets:
            bbox = det[:4]
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[2])
            y_max = int(bbox[3])

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

            annotated_img = cv2.rectangle(
                img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1
            )
    else:
        annotated_img = img
        cropped_image = img

    return annotated_img, cropped_image, ((x_min, y_min), (x_max, y_max))


def preprocess_face(img):
    resized = tf.image.resize_with_pad(img, 112, 112).numpy().astype(np.uint8)
    transposed = np.transpose(resized, (2, 0, 1))
    result = torch.from_numpy(transposed).unsqueeze(0).float()
    result.div_(255).sub_(0.5).div_(0.5)

    return result


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


target_width = 640
target_height = 640
roi_width = 200
roi_height = 200

# instantiate CenterFace
centerface = CenterFace(landmarks=True)

# load edgeface model
net = get_model("edgeface_xs_gamma_06", fp16=False)
net.load_state_dict(
    torch.load("checkpoints/edgeface_xs_gamma_06.pt", map_location="cpu")
)
net.eval()

# load classifier
classifier = joblib.load("svm_classifier.joblib")

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

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # draw roi box
    cv2.rectangle(
        frame,
        (top_left_x, top_left_y),
        (bottom_right_x, bottom_right_y),
        (255, 255, 0),
        2,
    )

    rect_img = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
    (
        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x],
        cropped_face,
        points,
    ) = detect_face(rect_img)

    resized_face = preprocess_face(
        cropped_face
    )  # resizing the cropped face to 112 * 112 as required by edgeface

    face_desc = inference(resized_face, net)

    pred = classifier.predict_proba(face_desc)

    confidence = np.max(pred)

    if confidence > 0.8:
        predicted_name = names[np.argmax(pred)]
    else:
        predicted_name = "Unknown"
        confidence = 0.0

    # logging.info(f"{predicted_name, pred}")
    cv2.putText(
        frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x],
        f"{predicted_name}: {confidence:.2%}",
        points[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2,
    )

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

    # frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)))
    cv2.imshow("face detection demo", frame[..., ::-1])
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
