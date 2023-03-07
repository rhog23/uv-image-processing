from skimage.morphology import skeletonize, thin
from skimage.util import invert
from skimage import data, io, filters
import matplotlib.pyplot as plt
import cv2

img = invert(data.horse())
# img = invert(io.imread("images/bw-objects.jpg", as_gray=True))
# img = filters.gaussian(img, 0.7)
# thresh = filters.threshold_otsu(img)
skeleton = skeletonize(img)
thinned = thin(img)
thinned_partial = thin(img, max_num_iter=20)
fig, axs = plt.subplots(nrows=2, ncols=2)
ax = axs.ravel()
ax[0].imshow(img, cmap="gray")
ax[0].axis("off")
ax[0].set_title("Original")
ax[1].imshow(skeleton, cmap="gray")
ax[1].axis("off")
ax[1].set_title("Skeleton")
ax[2].imshow(thinned, cmap="gray")
ax[2].axis("off")
ax[2].set_title("Thinned")
ax[3].imshow(thinned_partial, cmap="gray")
ax[3].axis("off")
ax[3].set_title("Partially Thinned")
plt.show()
