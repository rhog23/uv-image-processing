import cv2
import matplotlib.pyplot as plt
import numpy as np


def adjust_gamma(image, gamma=1.0):
    invGamma = 1 / gamma
    table = np.array([((i / 255) ** invGamma) * 255 for i in np.arange(0, 256)])

    return cv2.LUT(image, table)


img_dir = "../images/1.png"

img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

adjusted = adjust_gamma(img, gamma=1.5)

fig, axs = plt.subplots(ncols=3)
axs[0].set_title("Original Image")
axs[0].set_axis_off()
axs[0].imshow(img)
axs[1].set_title("Result")
axs[1].set_axis_off()
axs[1].imshow(np.hstack([img, adjusted]))
plt.show()
