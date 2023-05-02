import numpy as np
from skimage import io, color, util
import matplotlib.pyplot as plt
from scipy import ndimage

img = io.imread("images/1.png")
gray = color.rgb2gray(img)
filter_1 = np.divide([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 9)
filter_2 = np.divide([[1, 1, 1], [1, 2, 1], [1, 1, 1]], 10)
filter_3 = np.divide([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 16)
gauss_noise = util.random_noise(gray, mode="gaussian", mean=0.1, var=8e-2)
denoise_result_1 = ndimage.correlate(gray, filter_1)
denoise_result_2 = ndimage.correlate(gray, filter_2)
denoise_result_3 = ndimage.correlate(gray, filter_3)

fig, axs = plt.subplots(ncols=5)
axs[0].set_title("Gray Image")
axs[0].imshow(gray, cmap="gray")
axs[0].set_axis_off()
axs[1].set_title("Gaussian Noise")
axs[1].imshow(gauss_noise, cmap="gray")
axs[1].set_axis_off()
axs[2].set_title("Denoising 1")
axs[2].imshow(denoise_result_1, cmap="gray")
axs[2].set_axis_off()
axs[3].set_title("Denoising 2")
axs[3].imshow(denoise_result_2, cmap="gray")
axs[3].set_axis_off()
axs[4].set_title("Denoising 3")
axs[4].imshow(denoise_result_3, cmap="gray")
axs[4].set_axis_off()
plt.tight_layout()
plt.show()
