import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("images/bw-objects.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.array(([1, 1, 1], [1, 1, 1], [-1, -1, -1]), dtype="int")


result = cv2.morphologyEx(img, cv2.MORPH_HITMISS, kernel)

fig, axs = plt.subplots(ncols=2)
ax = axs.ravel()
ax[0].imshow(img, cmap="gray")
ax[0].axis("off")
ax[0].set_title("Original")
ax[1].imshow(result, cmap="gray")
ax[1].axis("off")
ax[1].set_title("Result")
plt.show()
