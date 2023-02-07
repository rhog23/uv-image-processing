from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

img_dir = "../images/1-gray.png"

img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
pil_img = Image.open(img_dir)

new_img_1 = cv2.bitwise_not(img)
new_img_2 = ImageOps.invert(pil_img)

fig, axs = plt.subplots(ncols=3)
axs[0].set_title("Original Image")
axs[0].set_axis_off()
axs[0].imshow(img)
axs[1].set_title("OpenCV Result")
axs[1].set_axis_off()
axs[1].imshow(new_img_1, cmap="gray")
axs[2].set_title("Pillow Result")
axs[2].set_axis_off()
axs[2].imshow(new_img_2, cmap="gray")
plt.show()
