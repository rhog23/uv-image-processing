import gradio as gr
import cv2
import numpy as np


# Fungsi deteksi tepi dan garis
def detect_edges_and_lines(image, edge_method):
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Deteksi tepi berdasarkan metode yang dipilih
    if edge_method == "Sobel":
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.uint8(edges / edges.max() * 255)
    elif edge_method == "Prewitt":
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(blur, -1, kernel_x)
        prewitt_y = cv2.filter2D(blur, -1, kernel_y)
        edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
        edges = np.uint8(edges / edges.max() * 255)
    else:  # Canny
        edges = cv2.Canny(blur, 50, 150)

    # Deteksi garis dengan Hough Transform
    lines_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Konversi ke RGB untuk Gradio
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    lines_rgb = cv2.cvtColor(lines_image, cv2.COLOR_BGR2RGB)

    return edges_rgb, lines_rgb


# Antarmuka Gradio
with gr.Blocks(title="Deteksi Garis dan Tepi") as demo:
    gr.Markdown("# Modul Deteksi Garis dan Tepi")
    gr.Markdown(
        "Unggah gambar untuk mendeteksi tepi dan garis dengan berbagai operator."
    )

    with gr.Row():
        with gr.Column():
            # Input
            image_input = gr.Image(label="Unggah Gambar", type="numpy")
            edge_method = gr.Radio(
                choices=["Sobel", "Prewitt", "Canny"],
                value="Canny",
                label="Pilih Operator Deteksi Tepi",
            )
            submit_btn = gr.Button("Proses")

        with gr.Column():
            # Output
            edges_output = gr.Image(label="Hasil Deteksi Tepi")
            lines_output = gr.Image(label="Hasil Deteksi Garis (Hough Transform)")

    # Event handler
    submit_btn.click(
        fn=detect_edges_and_lines,
        inputs=[image_input, edge_method],
        outputs=[edges_output, lines_output],
    )

# Jalankan aplikasi
demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
