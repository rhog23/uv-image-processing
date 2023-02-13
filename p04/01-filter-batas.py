from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

img = Image.open("../images/boneka2.tif")

filter_size = 3

min_img = img.filter(ImageFilter.MinFilter(size=filter_size))
max_img = img.filter(ImageFilter.MaxFilter(size=filter_size))

fig, axs = plt.subplots(ncols=3)
axs[0].set_title("Original Image")
axs[0].set_axis_off()
axs[0].imshow(img, cmap="gray")
axs[1].set_title("Min Image")
axs[1].set_axis_off()
axs[1].imshow(min_img, cmap="gray")
axs[2].set_title("Max Image")
axs[2].set_axis_off()
axs[2].imshow(max_img, cmap="gray")
plt.show()
