import cv2
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

img_dir = "../images/1.png"

# opencv
imcv = cv2.imread(img_dir)
imcv = cv2.cvtColor(imcv, cv2.COLOR_BGR2RGB)
# pil
impil = Image.open(img_dir)

# input
try:
    alpha = float(input("Enter alpha value: "))
    beta = int(input("Enter beta value: "))
    br_factor = float(input("Enter brightness factor: "))
    ct_factor = float(input("Enter contrast factor: "))

    rescv = cv2.convertScaleAbs(imcv, alpha=alpha, beta=beta)

    br_enhancer = ImageEnhance.Brightness(impil)
    ct_enhancer = ImageEnhance.Contrast(impil)
    respil = ct_enhancer.enhance(ct_factor)
    respil = br_enhancer.enhance(br_factor)

    fig, axs = plt.subplots(ncols=3)
    axs[0].set_title("Original Image")
    axs[0].set_axis_off()
    axs[0].imshow(impil)
    axs[1].set_title(f"OpenCv | α: {alpha} | β: {beta}")
    axs[1].set_axis_off()
    axs[1].imshow(rescv)
    axs[2].set_title(
        f"Pillow\nbrightness factor: {br_factor}\ncontrast factor: {ct_factor}"
    )
    axs[2].set_axis_off()
    axs[2].imshow(respil)
    plt.show()
except ValueError:
    print("Error, not a number")
