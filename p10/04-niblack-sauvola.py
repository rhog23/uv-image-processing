import matplotlib.pyplot as plt
from skimage import io, color, filters

img = io.imread("images/jahe.jpg")
gray = color.rgb2gray(img)
window_size = 909
thresh_niblack = filters.threshold_niblack(gray, window_size=window_size, k=1.5)
thresh_sauvola = filters.threshold_sauvola(gray, window_size=window_size)

binary_niblack = gray > thresh_niblack
binary_sauvola = gray > thresh_sauvola

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Gray")
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(binary_niblack, cmap="gray")
plt.title("Niblack")
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(binary_sauvola, cmap="gray")
plt.title("Sauvola")
plt.axis("off")
plt.show()
