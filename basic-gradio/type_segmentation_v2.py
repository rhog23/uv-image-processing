import cv2
import gradio as gr
from skimage import filters
import numpy as np


# Fungsi untuk memproses frame dengan metode segmentasi tertentu
def process_frame(frame, method):
    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Pilih metode segmentasi berdasarkan input dropdown
    if method == "Global Thresholding (T=127)":
        _, segmented = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    elif method == "Otsu's Thresholding":
        _, segmented = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    elif method == "Li's Thresholding":
        thresh_li = filters.threshold_li(gray)
        segmented = (gray > thresh_li).astype(np.uint8) * 255
    elif method == "Yen's Thresholding":
        thresh_yen = filters.threshold_yen(gray)
        segmented = (gray > thresh_yen).astype(np.uint8) * 255
    elif method == "Triangle Thresholding":
        thresh_tri = filters.threshold_triangle(gray)
        segmented = (gray > thresh_tri).astype(np.uint8) * 255
    else:
        segmented = (
            gray  # Default: tampilkan grayscale jika tidak ada metode yang cocok
        )

    # Konversi hasil ke format RGB untuk ditampilkan di Gradio
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_GRAY2RGB)

    # Kembalikan frame asli dan hasil segmentasi
    return frame, segmented_rgb


# Fungsi untuk streaming video dari webcam
def video_stream(method):
    # Buka webcam (0 adalah kamera default)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:
        # Baca frame dari webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi frame dari BGR (OpenCV) ke RGB (Gradio)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Proses frame dengan metode segmentasi yang dipilih
        original, segmented = process_frame(frame_rgb, method)

        # Kirim frame asli dan hasil segmentasi ke Gradio
        yield original, segmented

    # Lepaskan webcam setelah selesai
    cap.release()


# Buat antarmuka Gradio
interface = gr.Interface(
    fn=video_stream,  # Fungsi yang mengaktifkan webcam dan memproses stream
    inputs=gr.Dropdown(
        choices=[
            "Global Thresholding (T=127)",
            "Otsu's Thresholding",
            "Li's Thresholding",
            "Yen's Thresholding",
            "Triangle Thresholding",
        ],
        label="Pilih Metode Segmentasi",
    ),  # Dropdown untuk memilih metode segmentasi
    outputs=[
        gr.Image(label="Frame Asli", streaming=True),
        gr.Image(label="Hasil Segmentasi", streaming=True),
    ],
    live=True,  # Aktifkan pembaruan real-time
    title="Segmentasi Real-time dengan Pilihan Metode",
    description="Pilih metode segmentasi dari dropdown untuk membandingkannya dengan frame asli dari webcam.",
)

# Jalankan aplikasi
interface.launch(share=False, server_name="0.0.0.0", server_port=7860)
