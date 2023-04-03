import matplotlib.pyplot as plt
from skimage import io, color, segmentation

img = io.imread("images/jahe.jpg")
img_gry = color.rgb2gray(img)
cv = segmentation.chan_vese(
    img_gry,
    mu=.1,
    lambda1=1,
    lambda2=1,
    tol=1e-3,
    max_num_iter=200,
    dt=0.5,
    init_level_set="checkerboard",
    extended_output=True,
)

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
ax = axs.flatten()

ax[0].imshow(img_gry, cmap="gray")
ax[0].set_title("Original Image", fontsize=16)
ax[1].imshow(cv[0], cmap="gray")
ax[1].set_title(f"Chan-Vese segmentation - {len(cv[2])} iterations", fontsize=16)
ax[2].imshow(cv[1], cmap="gray")
ax[2].set_title(f"Final Level Set", fontsize=16)
for i in range(4):
    ax[i].set_axis_off()
plt.show()
