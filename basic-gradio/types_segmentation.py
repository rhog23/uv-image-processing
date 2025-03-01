import cv2
import gradio as gr
from skimage import filters
import numpy as np


# Fungsi untuk memproses frame dengan berbagai metode thresholding
def process_frame(frame):
    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Global Thresholding (T=127)
    _, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Otsu's Thresholding
    _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # Li's Thresholding
    thresh_li = filters.threshold_li(gray)
    binary_li = (gray > thresh_li).astype(np.uint8) * 255

    # Yen's Thresholding
    thresh_yen = filters.threshold_yen(gray)
    binary_yen = (gray > thresh_yen).astype(np.uint8) * 255

    # Triangle Thresholding
    thresh_tri = filters.threshold_triangle(gray)
    binary_tri = (gray > thresh_tri).astype(np.uint8) * 255

    # Konversi hasil ke format RGB untuk ditampilkan di Gradio
    binary_thresh_rgb = cv2.cvtColor(binary_thresh, cv2.COLOR_GRAY2RGB)
    otsu_thresh_rgb = cv2.cvtColor(otsu_thresh, cv2.COLOR_GRAY2RGB)
    binary_li_rgb = cv2.cvtColor(binary_li, cv2.COLOR_GRAY2RGB)
    binary_yen_rgb = cv2.cvtColor(binary_yen, cv2.COLOR_GRAY2RGB)
    binary_tri_rgb = cv2.cvtColor(binary_tri, cv2.COLOR_GRAY2RGB)

    # Kembalikan frame asli dan hasil thresholding
    return (
        frame,
        binary_thresh_rgb,
        otsu_thresh_rgb,
        binary_li_rgb,
        binary_yen_rgb,
        binary_tri_rgb,
    )


# Fungsi untuk streaming video dari webcam menggunakan OpenCV
def video_stream():
    # Buka webcam (0 adalah kamera default)
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi frame dari BGR (OpenCV) ke RGB (Gradio)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses frame dengan berbagai metode thresholding
        original, binary_thresh, otsu_thresh, binary_li, binary_yen, binary_tri = (
            process_frame(frame_rgb)
        )

        # Kirim frame asli dan hasil thresholding ke Gradio
        yield original, binary_thresh, otsu_thresh, binary_li, binary_yen, binary_tri

    # Lepaskan webcam setelah selesai
    cap.release()


# Buat antarmuka Gradio
interface = gr.Interface(
    fn=video_stream,  # Fungsi yang mengaktifkan webcam dan memproses stream
    inputs=None,  # Tidak ada input manual, webcam diatur di fungsi
    outputs=[
        gr.Image(label="Frame Asli", streaming=True),
        gr.Image(label="Global Thresholding (T=127)", streaming=True),
        gr.Image(label="Otsu's Thresholding", streaming=True),
        gr.Image(label="Li's Thresholding", streaming=True),
        gr.Image(label="Yen's Thresholding", streaming=True),
        gr.Image(label="Triangle Thresholding", streaming=True),
    ],
    live=True,  # Aktifkan pembaruan real-time
    title="Perbandingan Thresholding Real-time",
    description="Menampilkan frame asli dari webcam dan hasil segmentasi dengan berbagai metode thresholding dalam format grid.",
)

# Jalankan aplikasi
interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
