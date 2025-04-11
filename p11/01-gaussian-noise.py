from skimage import io, color, util
import matplotlib.pyplot as plt

image = io.imread("images/1.png")
gray = color.rgb2gray(image)
mean = [0.01, 0.7]
variance = [0.01, 1]
result_1 = util.random_noise(gray, mode="gaussian", mean=mean[0], var=variance[0])
result_2 = util.random_noise(gray, mode="gaussian", mean=mean[1], var=variance[1])

fig, axs = plt.subplots(ncols=2)
axs[0].set_title(f"Mean: {mean[0]} | Variance: {variance[0]}")
axs[0].imshow(result_1, cmap="grey")
axs[0].set_axis_off()
axs[1].set_title(f"Mean: {mean[1]} | Variance: {mean[1]}")
axs[1].imshow(result_2)
axs[1].set_axis_off()
plt.tight_layout()
plt.show()
