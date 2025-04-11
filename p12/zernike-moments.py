import gradio as gr
import numpy as np
import mahotas as mh
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

def process_image(image):
    # Convert PIL image to numpy array (grayscale)
    img_array = np.array(image.convert("L"))

    # Step 1: Convert to Binary Image
    threshold = mh.thresholding.otsu(img_array)
    binary_img = (img_array > threshold).astype(np.uint8)

    # Save binary image for display
    plt.imshow(binary_img, cmap='gray')
    plt.title("Binary Image")
    plt.axis('off')
    plt.savefig("binary_image.png")
    plt.close()

    # Step 2: Scaling Normalization
    # Compute the bounding box of the object
    coords = np.where(binary_img > 0)
    if len(coords[0]) == 0:  # Handle case where no object is detected
        return "No object detected in image.", None, None
    
    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()
    obj_height = max_y - min_y + 1
    obj_width = max_x - min_x + 1
    scale_factor = 40 / max(obj_height, obj_width)  # Scale to fit within radius=40
    scaled_img = mh.imresize(binary_img, scale_factor)

    # Pad to a square image (e.g., 64x64)
    target_size = 64
    padded_img = np.zeros((target_size, target_size), dtype=np.uint8)
    y_offset = (target_size - scaled_img.shape[0]) // 2
    x_offset = (target_size - scaled_img.shape[1]) // 2
    padded_img[y_offset:y_offset+scaled_img.shape[0], x_offset:x_offset+scaled_img.shape[1]] = scaled_img

    # Step 3: Translation Normalization
    centroid_y, centroid_x = ndimage.measurements.center_of_mass(padded_img)
    shift_y = target_size // 2 - int(centroid_y)
    shift_x = target_size // 2 - int(centroid_x)
    centered_img = ndimage.shift(padded_img, (shift_y, shift_x), mode='constant', cval=0)

    # Save centered image for display
    plt.imshow(centered_img, cmap='gray')
    plt.title("Centered Binary Image")
    plt.axis('off')
    plt.savefig("centered_image.png")
    plt.close()

    # Step 4: Calculate ZMD
    radius = 21  # Radius for Zernike moments
    degree = 8   # Maximum order of Zernike moments
    zmd = mh.features.zernike_moments(centered_img, radius=radius, degree=degree)

    # Step 5: Normalize ZMD (by Z_0,0)
    zmd_normalized = zmd / zmd[0] if zmd[0] != 0 else zmd

    # Prepare text output
    output_text = "Raw Zernike Moments:\n"
    for i, val in enumerate(zmd):
        output_text += f"Z_{i}: {val:.4f}\n"
    output_text += "\nNormalized Zernike Moments:\n"
    for i, val in enumerate(zmd_normalized):
        output_text += f"Z_{i}: {val:.4f}\n"

    return output_text, "binary_image.png", "centered_image.png"

# Define Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=[
        gr.Textbox(label="Zernike Moments (Raw and Normalized)"),
        gr.Image(label="Binary Image"),
        gr.Image(label="Centered Binary Image")
    ],
    title="Zernike Moments Calculator",
    description="Upload an image to compute its Zernike moments. Displays the binary image, centered binary image, and raw/normalized moments."
)

# Launch the app
interface.launch()