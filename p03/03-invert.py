from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

img_dir = "../images/treehouse_5.jpeg"  # lokasi citra disesuaikan

# opencv
img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Pillow
pil_img = Image.open(img_dir)

rescv = cv2.bitwise_not(img)
respil = ImageOps.invert(pil_img)

fig, axs = plt.subplots(ncols=3)
axs[0].set_title("Original Image")
axs[0].set_axis_off()
axs[0].imshow(img)
axs[1].set_title("OpenCV Result")
axs[1].set_axis_off()
axs[1].imshow(rescv)
axs[2].set_title("Pillow Result")
axs[2].set_axis_off()
axs[2].imshow(respil)
plt.show()
