import cv2
import numpy as np
import gradio as gr

# Load classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
upper_body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_upperbody.xml"
)
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def process_frame(frame):
    # Convert RGB to BGR (OpenCV expects BGR format)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Downscale frame for faster processing
    scale_percent = 50
    width = int(frame_bgr.shape[1] * scale_percent / 100)
    height = int(frame_bgr.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_frame = cv2.resize(frame_bgr, dim)
    small_gray = cv2.resize(gray, dim)

    # Detect faces with Haar Cascade
    faces = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Detect upper bodies with Haar Cascade
    upper_bodies = upper_body_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    # Detect full bodies with HOG
    bodies, _ = hog.detectMultiScale(
        small_frame, winStride=(4, 4), padding=(8, 8), scale=1.05
    )

    # Scale bounding boxes back to original size
    scale_x = frame_bgr.shape[1] / width
    scale_y = frame_bgr.shape[0] / height

    # Draw detections
    for x, y, w, h in faces:
        frame_bgr = cv2.rectangle(
            frame_bgr,
            (int(x * scale_x), int(y * scale_y)),
            (int((x + w) * scale_x), int((y + h) * scale_y)),
            (0, 255, 0),
            2,
        )

    for x, y, w, h in upper_bodies:
        frame_bgr = cv2.rectangle(
            frame_bgr,
            (int(x * scale_x), int(y * scale_y)),
            (int((x + w) * scale_x), int((y + h) * scale_y)),
            (255, 0, 0),
            2,
        )

    for x, y, w, h in bodies:
        frame_bgr = cv2.rectangle(
            frame_bgr,
            (int(x * scale_x), int(y * scale_y)),
            (int((x + w) * scale_x), int((y + h) * scale_y)),
            (0, 0, 255),
            2,
        )

    # Convert back to RGB for Gradio
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# Create Gradio interface
demo = gr.Interface(
    fn=process_frame,
    inputs=gr.Image(source="webcam", streaming=True),
    outputs="image",
    title="Real-Time Face & Body Detection",
    description="Detection using traditional CV techniques (Haar Cascades & HOG)",
    live=True,
)

# Launch the app
if __name__ == "__main__":
    demo.launch()
