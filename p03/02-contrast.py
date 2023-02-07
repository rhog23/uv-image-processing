from PIL import Image, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import time

img_dir = "../images/1.png"  # lokasi citra disesuaikan

img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pil = Image.open(img_dir)

try:
    alpha = float(input("Enter alpha (contrast): "))
    beta = int(input("Enter beta (brightness): "))
    factor = float(input("Enter factor (Pillow): "))
except ValueError:
    print("Error, not a number")

new_img_1 = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
enhancer = ImageEnhance.Contrast(img_pil)
new_img_2 = enhancer.enhance(factor)

fig, axs = plt.subplots(ncols=3, figsize=(7, 4))
axs[0].set_title("Original Image")
axs[0].set_axis_off()
axs[0].imshow(img)
axs[1].set_title(f"Contrast OpenCV\nAlpha: {alpha}")
axs[1].set_axis_off()
axs[1].imshow(new_img_1)
axs[2].set_title(f"Contrast Pillow\nFactor: {factor}")
axs[2].set_axis_off()
axs[2].imshow(new_img_2)

plt.show()
