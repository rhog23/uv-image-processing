import cv2
import gradio as gr
import numpy as np


def process_frame(frame):
    """Fungsi untuk menerapkan pemrosesan citra (dalam kasus ini segmentasi menggunakan metode Otsu)

    Args:
        frame: video frame

    Returns:
        hasil segmentasi
    """
    # Konversi warna dari RGB ke Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Menerapkan Otsu
    _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)

    return segmented_rgb


def video_stream():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV menggunakan susunan BGR, sedangkan Gradio menggunakan susunan RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Memproses setiap frame untuk menampilkan hasil segmentasi
        result = process_frame(frame_rgb)

        yield frame_rgb, result

        # return frame_rgb, result

    cap.release()


interface = gr.Interface(
    fn=video_stream,
    inputs=None,
    outputs=[
        gr.Image(label="Video Asli", streaming=True),
        gr.Image(label="Video Hasil", streaming=True),
    ],
    live=True,
    title="Otsu Segmentation Webcam App",
    description="Real-time video segmentation using Otsu's method.",
)

interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
