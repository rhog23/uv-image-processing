import gradio as gr
import numpy as np
from skimage import color, util
from scipy import ndimage
from PIL import Image


def add_noise_and_denoise(
    image,
    noise_type,
    grayscale_option,
    gaussian_variance=0.01,
    sp_amount=0.05,
    salt_pepper_ratio=0.5,
    denoise_option="None",
):
    """
    Fungsi untuk menambahkan noise dan menghilangkan noise dari citra.

    Parameters:
    - image: Citra input (PIL Image)
    - noise_type: Jenis noise ('gaussian', 'salt', 'pepper', 's&p')
    - grayscale_option: Pilihan untuk mengubah ke grayscale ('yes' atau 'no')
    - gaussian_variance: Variansi untuk Gaussian noise
    - sp_amount: Persentase piksel yang terkena Salt and Pepper noise
    - salt_pepper_ratio: Rasio antara salt dan pepper
    - denoise_option: Pilihan denoising ('None', 'Low-Pass 1', 'Low-Pass 2', 'Low-Pass 3', 'Median')

    Returns:
    - Citra dengan noise dan citra setelah denoising (PIL Image)
    """
    # Konversi PIL Image ke array numpy
    image_np = np.array(image)

    # Jika citra adalah RGB, tetap gunakan RGB; jika grayscale, konversi
    is_grayscale = grayscale_option == "yes"

    if is_grayscale:
        if len(image_np.shape) == 3 and image_np.shape[2] in [3, 4]:
            image_np = color.rgb2gray(image_np)
        elif len(image_np.shape) == 2:
            pass
        else:
            raise ValueError("Format citra tidak didukung.")
    else:
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        elif image_np.shape[2] == 4:
            image_np = image_np[:, :, :3]

    # Tambahkan noise
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

    # Simpan citra dengan noise untuk output
    noisy_image_output = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
    if is_grayscale and len(noisy_image_output.shape) == 2:
        noisy_image_output = np.stack([noisy_image_output] * 3, axis=-1)

    # Terapkan denoising
    if denoise_option == "None":
        denoised_image = noisy_image
    elif denoise_option == "Low-Pass 1":
        kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9.0
        if is_grayscale:
            denoised_image = ndimage.correlate(noisy_image, kernel, mode="nearest")
        else:
            denoised_image = np.stack(
                [
                    ndimage.correlate(noisy_image[:, :, i], kernel, mode="nearest")
                    for i in range(3)
                ],
                axis=-1,
            )
    elif denoise_option == "Low-Pass 2":
        kernel = np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]) / 10.0
        if is_grayscale:
            denoised_image = ndimage.correlate(noisy_image, kernel, mode="nearest")
        else:
            denoised_image = np.stack(
                [
                    ndimage.correlate(noisy_image[:, :, i], kernel, mode="nearest")
                    for i in range(3)
                ],
                axis=-1,
            )
    elif denoise_option == "Low-Pass 3":
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        if is_grayscale:
            denoised_image = ndimage.correlate(noisy_image, kernel, mode="nearest")
        else:
            denoised_image = np.stack(
                [
                    ndimage.correlate(noisy_image[:, :, i], kernel, mode="nearest")
                    for i in range(3)
                ],
                axis=-1,
            )
    elif denoise_option == "Median":
        if is_grayscale:
            denoised_image = ndimage.median_filter(noisy_image, size=3)
        else:
            denoised_image = np.stack(
                [ndimage.median_filter(noisy_image[:, :, i], size=3) for i in range(3)],
                axis=-1,
            )

    # Konversi citra denoised ke uint8
    denoised_image = np.clip(denoised_image * 255, 0, 255).astype(np.uint8)
    if is_grayscale and len(denoised_image.shape) == 2:
        denoised_image = np.stack([denoised_image] * 3, axis=-1)

    return Image.fromarray(noisy_image_output), Image.fromarray(denoised_image)


# Fungsi untuk antarmuka Gradio
def gradio_interface(
    image,
    noise_type,
    grayscale_option,
    gaussian_variance,
    sp_amount,
    salt_pepper_ratio,
    denoise_option,
):
    """
    Fungsi antarmuka Gradio untuk menampilkan citra dengan noise dan hasil denoising.
    """
    if image is None:
        return None, None, "Silakan unggah citra terlebih dahulu."

    # Proses citra
    noisy_image, denoised_image = add_noise_and_denoise(
        image,
        noise_type,
        grayscale_option,
        gaussian_variance,
        sp_amount,
        salt_pepper_ratio,
        denoise_option,
    )

    return noisy_image, denoised_image, "Citra berhasil diproses! 🙌🥳"


# Buat antarmuka Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Aplikasi Penambahan dan Penghilangan Noise pada Citra Digital")
    gr.Markdown(
        "Unggah citra, pilih apakah akan diubah ke grayscale, pilih jenis noise, atur parameter, dan pilih metode denoising untuk melihat hasilnya 😀."
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Unggah Citra")
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
            denoise_option = gr.Dropdown(
                choices=["None", "Low-Pass 1", "Low-Pass 2", "Low-Pass 3", "Median"],
                label="Metode Denoising",
                value="None",
            )
            apply_button = gr.Button("Terapkan Noise dan Denoising")

        with gr.Column():
            noisy_image_output = gr.Image(label="Citra dengan Noise")
            denoised_image_output = gr.Image(label="Citra setelah Denoising")
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
            denoise_option,
        ],
        outputs=[noisy_image_output, denoised_image_output, output_text],
    )

# Luncurkan aplikasi
demo.launch()
