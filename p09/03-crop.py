import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature

image = io.imread("images/jahe.jpg")
image_gray = color.rgb2gray(image)
image_gray = filters.gaussian(image_gray, sigma=5)
edge_canny = feature.canny(image_gray, sigma=2)

pts = np.argwhere(edge_canny > 0)
y1, x1 = pts.min(axis=0)
y2, x2 = pts.max(axis=0)
# cropped = image[y1:y2, x1:x2]
detected = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (114, 9, 183), 3)
fig, axs = plt.subplots(nrows=2, ncols=2)
axs = axs.ravel()
axs[0].imshow(image)
axs[1].imshow(image_gray, cmap="gray")
axs[2].imshow(edge_canny)
axs[3].imshow(detected)
for ax in axs:
    ax.set_axis_off()
plt.show()
