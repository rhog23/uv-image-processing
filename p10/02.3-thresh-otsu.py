import matplotlib.pyplot as plt
from skimage import io, color, filters

img = io.imread("images/jahe.jpg")
gray = color.rgb2gray(img)

# threshold otsu
thresh_otsu = filters.threshold_otsu(gray)
print(thresh_otsu)
binary_otsu = gray > thresh_otsu

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original")
axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Gray")
axes[1, 0].imshow(binary_otsu, cmap="gray")
axes[1, 0].set_title("Otsu Threshold Result")
axes[1, 1].hist(gray.ravel(), bins=256)
axes[1, 1].axvline(thresh_otsu, color="r")
plt.show()
