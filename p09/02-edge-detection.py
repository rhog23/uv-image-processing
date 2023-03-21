import matplotlib.pyplot as plt
from skimage import io, color, filters, feature

image = io.imread("images/mildlee-8N6z4yXUkwY-unsplash.jpg")
image_gray = color.rgb2gray(image)
edge_roberts = filters.roberts(image_gray)
edge_sobel = filters.sobel(image_gray)
edge_prewitt = filters.prewitt(image_gray)
edge_scharr = filters.scharr(image_gray)
edge_canny = feature.canny(image_gray, sigma=2)

fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(10, 10))
ax = axs.ravel()
ax[0].imshow(image)
ax[0].set_title("Original Image", fontsize=14)
ax[1].imshow(edge_roberts, cmap="gray")
ax[1].set_title("Roberts Edge Detection", fontsize=14)
ax[2].imshow(edge_sobel, cmap="gray")
ax[2].set_title("Sobel Edge Detection", fontsize=14)
ax[3].imshow(edge_prewitt, cmap="gray")
ax[3].set_title("Prewitt Edge Detection", fontsize=14)
ax[4].imshow(edge_scharr, cmap="gray")
ax[4].set_title("Scharr Edge Detection", fontsize=14)
ax[5].imshow(edge_canny, cmap="gray")
ax[5].set_title("Canny Edge Detection", fontsize=14)
for i, _ in enumerate(ax):
    ax[i].set_axis_off()
plt.tight_layout()
plt.show()
