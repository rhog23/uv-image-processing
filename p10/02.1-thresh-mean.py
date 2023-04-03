import matplotlib.pyplot as plt
from skimage import io, color, filters

img = io.imread("images/jahe.jpg")
gray = color.rgb2gray(img)

# threshold mean
thresh_mean = filters.threshold_mean(gray)
binary_mean = gray > thresh_mean

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
axes[0, 0].imshow(img)
axes[0, 0].set_title("Original")
axes[0, 1].imshow(gray, cmap="gray")
axes[0, 1].set_title("Gray")
axes[1, 0].imshow(binary_mean, cmap="gray")
axes[1, 0].set_title("Mean Threshold Result")
axes[1, 1].hist(gray.ravel(), bins=256)
axes[1, 1].axvline(thresh_mean, color="r")
plt.show()
