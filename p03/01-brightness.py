from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

img_dir = "../images/1.png"  # lokasi citra disesuaikan dengan yang dimiliki
img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
new_image = np.zeros(img.shape, img.dtype)
new_image_2 = np.zeros(img.shape, img.dtype)

try:
    alpha = float(input("Enter a value for alpha (contrast): "))
    beta = int(input("Enter a value for beta (brightness): "))
    factor = float(input("Enter Pillow brightness factor: "))
except ValueError:
    print("Error, not a number")

start = time.time()
for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        for c in range(img.shape[2]):
            new_image[y, x, c] = np.clip(alpha * img[y, x, c] + beta, 0, 255)
end = time.time()

print(f"Processing time (manual process): {end-start:.5f} s")

start = time.time()
new_image_2 = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
end = time.time()

print(f"Processing time (opencv function): {end-start:.5f} s")

# PIL
pil_img = Image.open(img_dir)
start = time.time()
enhancer = ImageEnhance.Brightness(pil_img)
pil_output = enhancer.enhance(factor)
end = time.time()
print(f"Processing time (PIL): {end-start:.5f} s")

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
axs[0, 0].set_title("Original Image")
axs[0, 0].imshow(img)
axs[0, 0].set_axis_off()
axs[0, 1].set_title(f"Image Brightness (Manual)\nAlpha:{alpha} | Beta:{beta}")
axs[0, 1].imshow(new_image)
axs[0, 1].set_axis_off()
axs[1, 0].set_title(f"Image Brightness OpenCV\nAlpha:{alpha} | Beta:{beta}")
axs[1, 0].imshow(new_image_2)
axs[1, 0].set_axis_off()
axs[1, 1].set_title(f"Image Brightness Pillow\nFactor: {factor}")
axs[1, 1].imshow(pil_output, cmap="gray")
axs[1, 1].set_axis_off()
plt.show()
