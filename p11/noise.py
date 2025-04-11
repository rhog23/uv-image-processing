import gradio as gr
import numpy as np
from skimage import color, util
from PIL import Image


def add_noise(
    image,
    noise_type,
    grayscale_option,
    gaussian_variance=0.01,
    sp_amount=0.05,
    salt_pepper_ratio=0.5,
):
    """
    Fungsi untuk menambahkan noise ke citra, dengan opsi untuk mengubah ke grayscale.

    Parameters:
    - image: Citra input (PIL Image)
    - noise_type: Jenis noise ('gaussian', 'salt', 'pepper', 's&p')
    - grayscale_option: Pilihan untuk mengubah ke grayscale ('yes' atau 'no')
    - gaussian_variance: Variansi untuk Gaussian noise
    - sp_amount: Persentase piksel yang terkena Salt and Pepper noise
    - salt_pepper_ratio: Rasio antara salt dan pepper (0 hingga 1)

    Returns:
    - Citra dengan noise (PIL Image)
    """
    # Konversi PIL Image ke array numpy
    image_np = np.array(image)
    print("Nilai piksel sebelum noise:", image_np.min(), image_np.max())

    # Jika citra adalah RGB, tetap gunakan RGB; jika grayscale, konversi
    is_grayscale = grayscale_option == "yes"

    if is_grayscale:
        # Konversi ke grayscale
        if len(image_np.shape) == 3 and image_np.shape[2] in [3, 4]:
            image_np = color.rgb2gray(image_np)
        elif len(image_np.shape) == 2:
            pass  # Sudah grayscale
        else:
            raise ValueError("Format citra tidak didukung.")
    else:
        # Pastikan citra adalah RGB
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)

    # Tambahkan noise sesuai jenis
    if noise_type == "gaussian":
        noisy_image = util.random_noise(
            image_np, mode="gaussian", var=gaussian_variance, clip=True
        )
    elif noise_type == "salt":
        noisy_image = util.random_noise(
            image_np, mode="salt", amount=sp_amount, clip=True
        )
    elif noise_type == "pepper":
        noisy_image = util.random_noise(
            image_np, mode="pepper", amount=sp_amount, clip=True
        )
    elif noise_type == "s&p":
        noisy_image = util.random_noise(
            image_np,
            mode="s&p",
            amount=sp_amount,
            salt_vs_pepper=salt_pepper_ratio,
            clip=True,
        )
    else:
        noisy_image = image_np

    print("Nilai piksel setelah noise:", noisy_image.min(), noisy_image.max())

    # Kembalikan ke skala [0, 255] dan konversi ke uint8
    noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)

    # Jika grayscale, pastikan output memiliki 3 kanal untuk kompatibilitas Gradio
    if is_grayscale and len(noisy_image.shape) == 2:
        noisy_image = np.stack([noisy_image] * 3, axis=-1)

    # Konversi kembali ke PIL Image
    return Image.fromarray(noisy_image)


# Fungsi untuk antarmuka Gradio
def gradio_interface(
    image, noise_type, grayscale_option, gaussian_variance, sp_amount, salt_pepper_ratio
):
    """
    Fungsi antarmuka Gradio untuk menampilkan citra asli dan citra dengan noise.
    """
    if image is None:
        return None, "Silakan upload citra terlebih dahulu."

    # Tambahkan noise
    noisy_image = add_noise(
        image,
        noise_type,
        grayscale_option,
        gaussian_variance,
        sp_amount,
        salt_pepper_ratio,
    )

    return noisy_image, "Citra berhasil diproses! 🙌🥳"


# Buat antarmuka Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Aplikasi Penambahan Noise pada Citra Digital")
    gr.Markdown(
        "Upload citra, pilih apakah akan diubah ke grayscale, pilih jenis noise, dan atur parameter untuk melihat hasilnya 😀."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Citra")
            grayscale_option = gr.Radio(
                choices=["yes", "no"], label="Ubah ke Grayscale?", value="no"
            )
            noise_type = gr.Dropdown(
                choices=["gaussian", "salt", "pepper", "s&p"],
                label="Jenis Noise",
                value="gaussian",
            )
            gaussian_variance = gr.Slider(
                minimum=0.001,
                maximum=0.1,
                step=0.001,
                value=0.01,
                label="Variansi Gaussian Noise",
                visible=True,
            )
            sp_amount = gr.Slider(
                minimum=0.01,
                maximum=0.5,
                step=0.01,
                value=0.05,
                label="Jumlah Salt/Pepper Noise (%)",
                visible=False,
            )
            salt_pepper_ratio = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                step=0.1,
                value=0.5,
                label="Rasio Salt vs. Pepper",
                visible=False,
            )
            apply_button = gr.Button("Terapkan Noise")

        with gr.Column():
            output_image = gr.Image(label="Hasil Citra dengan Noise")
            output_text = gr.Textbox(label="Status")

    # Logika untuk mengatur visibilitas parameter berdasarkan jenis noise
    def update_visibility(noise_type):
        if noise_type == "gaussian":
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
            )
        else:
            return (
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=(noise_type == "s&p")),
            )

    noise_type.change(
        fn=update_visibility,
        inputs=noise_type,
        outputs=[gaussian_variance, sp_amount, salt_pepper_ratio],
    )

    # Hubungkan tombol dengan fungsi
    apply_button.click(
        fn=gradio_interface,
        inputs=[
            image_input,
            noise_type,
            grayscale_option,
            gaussian_variance,
            sp_amount,
            salt_pepper_ratio,
        ],
        outputs=[output_image, output_text],
    )

# Luncurkan aplikasi
demo.launch()
