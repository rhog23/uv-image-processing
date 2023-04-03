from skimage import io, color, util
import matplotlib.pyplot as plt

image = io.imread("images/1.png")
gray = color.rgb2gray(image)
poisson = util.random_noise(image)
poisson_gray = util.random_noise(gray)

fig, axs = plt.subplots(nrows=2, ncols=2)
ax = axs.flatten()
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[1].imshow(poisson)
ax[1].set_title("Poisson")
ax[2].imshow(gray)
ax[2].set_title("Original Image (Grayscale)")
ax[3].imshow(poisson_gray)
ax[3].set_title("Poisson (Grayscale)")
for a in ax:
    a.set_axis_off()
plt.tight_layout()
plt.show()
