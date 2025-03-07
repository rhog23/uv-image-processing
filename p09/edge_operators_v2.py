import gradio as gr
import numpy as np
from skimage import filters, feature, img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import gaussian, prewitt, laplace, roberts


# Fungsi deteksi tepi menggunakan scikit-image
def detect_edges(image, edge_operator):
    # Preprocessing: Konversi ke grayscale dan Gaussian Blur
    gray = rgb2gray(image)  # Scikit-image mengharapkan gambar dalam rentang [0, 1]
    blur = gaussian(gray, sigma=1)  # Sigma = 1 untuk blur ringan

    # Pilih operator deteksi tepi
    if edge_operator == "Sobel":
        edges = filters.sobel(blur)  # Sobel langsung dari scikit-image

    elif edge_operator == "Prewitt":
        edges = prewitt(blur)

    elif edge_operator == "Roberts":
        edges = roberts(blur)

    elif edge_operator == "Laplacian":
        edges = laplace(blur)  # Laplacian menghasilkan nilai negatif dan positif
        edges = np.abs(edges)  # Ambil nilai absolut untuk menghilangkan negatif

    else:  # Canny
        edges = feature.canny(blur, sigma=1, low_threshold=0.1, high_threshold=0.2)

    # Normalisasi dan konversi ke format uint8 untuk Gradio
    if edges.max() > 0:  # Cek untuk menghindari pembagian dengan nol
        edges = img_as_ubyte(edges / edges.max())
    else:
        edges = img_as_ubyte(edges)  # Jika max=0, langsung konversi tanpa normalisasi

    # Konversi ke RGB untuk ditampilkan di Gradio
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return edges_rgb


# Antarmuka Gradio
with gr.Blocks(title="Deteksi Tepi dengan Scikit-Image") as demo:
    gr.Markdown("# Deteksi Tepi dalam Pengolahan Citra (Scikit-Image)")
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
