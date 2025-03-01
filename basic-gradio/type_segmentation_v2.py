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
    yield frame, segmented_rgb


# Fungsi untuk streaming video dari webcam menggunakan cv2.VideoCapture
def video_stream(method):
    # Buka webcam (ganti indeks sesuai kamera Anda, misalnya 0 atau 1)
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

        # Proses frame dengan metode segmentasi yang dipilih
        original, segmented = process_frame(frame_rgb, method)

        # Kirim frame asli dan hasil segmentasi ke Gradio
        yield original, segmented

    # Lepaskan webcam setelah selesai
    cap.release()


# Buat antarmuka dengan gr.Blocks
with gr.Blocks(title="Segmentasi Real-time dengan Pilihan Metode") as demo:
    # Komponen dropdown untuk memilih metode segmentasi
    method_dropdown = gr.Dropdown(
        choices=[
            "Global Thresholding (T=127)",
            "Otsu's Thresholding",
            "Li's Thresholding",
            "Yen's Thresholding",
            "Triangle Thresholding",
        ],
        label="Pilih Metode Segmentasi",
        value="Otsu's Thresholding",  # Nilai awal
    )

    # Komponen untuk menampilkan frame asli dan hasil segmentasi
    with gr.Row():
        original_output = gr.Image(label="Frame Asli")
        segmented_output = gr.Image(label="Hasil Segmentasi")

    # Event handler: perbarui output saat dropdown berubah
    method_dropdown.change(
        fn=video_stream,
        inputs=method_dropdown,
        outputs=[original_output, segmented_output],
    )

# Jalankan aplikasi
demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
