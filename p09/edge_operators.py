import gradio as gr
import cv2
import numpy as np


# Fungsi deteksi tepi
def detect_edges(image, edge_operator):
    # Preprocessing: Konversi ke grayscale dan blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Pilih operator deteksi tepi
    if edge_operator == "Sobel":
        sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobel_x**2 + sobel_y**2)
        edges = np.uint8(edges / edges.max() * 255)

    elif edge_operator == "Prewitt":
        kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        prewitt_x = cv2.filter2D(blur, -1, kernel_x)
        prewitt_y = cv2.filter2D(blur, -1, kernel_y)
        edges = np.sqrt(prewitt_x**2 + prewitt_y**2)
        edges = np.uint8(edges / edges.max() * 255)

    elif edge_operator == "Roberts":
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        roberts_x = cv2.filter2D(blur, -1, kernel_x)
        roberts_y = cv2.filter2D(blur, -1, kernel_y)
        edges = np.abs(roberts_x) + np.abs(roberts_y)
        edges = np.uint8(edges / edges.max() * 255)

    elif edge_operator == "Laplacian":
        edges = cv2.Laplacian(blur, cv2.CV_64F)
        edges = np.uint8(np.absolute(edges) / np.absolute(edges).max() * 255)

    else:  # Canny
        edges = cv2.Canny(blur, 50, 150)

    # Konversi ke RGB untuk Gradio
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges_rgb


# Antarmuka Gradio
with gr.Blocks(title="Deteksi Tepi dengan Berbagai Operator") as demo:
    gr.Markdown("# Deteksi Tepi dalam Pengolahan Citra")
    gr.Markdown("Unggah gambar dan pilih operator deteksi tepi untuk melihat hasilnya.")

    with gr.Row():
        with gr.Column():
            # Input
            image_input = gr.Image(label="Unggah Gambar", type="numpy")
            edge_operator = gr.Dropdown(
                choices=["Sobel", "Prewitt", "Roberts", "Laplacian", "Canny"],
                value="Canny",
                label="Pilih Operator Deteksi Tepi",
            )
            submit_btn = gr.Button("Proses")

        with gr.Column():
            # Output
            edges_output = gr.Image(label="Hasil Deteksi Tepi")

    # Event handler
    submit_btn.click(
        fn=detect_edges, inputs=[image_input, edge_operator], outputs=edges_output
    )

# Jalankan aplikasi
demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
