import cv2
import gradio as gr
import numpy as np


# Function to process each frame
def process_frame(frame):
    # Convert RGB frame to grayscale (Otsu works on single-channel images)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply Otsu's thresholding
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert the segmented image back to RGB for display (Gradio expects RGB)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)

    return segmented_rgb


# Function to capture and process video stream
def video_stream():
    # Open the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV uses BGR, Gradio expects RGB, so convert it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with Otsu's method
        result = process_frame(frame_rgb)

        # Yield the processed frame for Gradio's live stream
        yield result

    cap.release()


# Create the Gradio interface
interface = gr.Interface(
    fn=video_stream,
    inputs=None,  # No manual input; we'll use the webcam stream
    outputs=gr.Image(streaming=True),  # Stream the processed video
    live=True,  # Live streaming
    title="Otsu Segmentation Webcam App",
    description="Real-time video segmentation using Otsu's method.",
)

# Launch the app, making it accessible over the local network
interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
