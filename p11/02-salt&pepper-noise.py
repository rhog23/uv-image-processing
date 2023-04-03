from skimage import io, color, util
import matplotlib.pyplot as plt

image = io.imread("images/1.png")
gray = color.rgb2gray(image)
sp = util.random_noise(image, mode="s&p")
sp_gray = util.random_noise(gray, mode="s&p")

fig, axs = plt.subplots(nrows=2, ncols=2)
ax = axs.flatten()
ax[0].imshow(image)
ax[0].set_title("RGB Image")
ax[1].imshow(sp)
ax[1].set_title("Salt&Pepper Result")
ax[2].imshow(gray, cmap="gray")
ax[2].set_title("Grayscale Image")
ax[3].imshow(sp_gray, cmap="gray")
ax[3].set_title("Salt&Pepper Result (Grayscale)")
for a in ax:
    a.set_axis_off()
plt.tight_layout()
plt.show()
