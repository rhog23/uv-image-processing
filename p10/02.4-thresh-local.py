import matplotlib.pyplot as plt
from skimage import io, color, filters

img = io.imread("images/jahe.jpg")
gray = color.rgb2gray(img)

# threshold local
block_size = 507
offset = 0.01
thresh_local = filters.threshold_local(gray, block_size, offset=offset)
binary_local = gray > thresh_local

fig, axes = plt.subplots(ncols=3, figsize=(8, 6))
axes[0].imshow(img)
axes[0].set_title("Original")
axes[1].imshow(gray, cmap="gray")
axes[1].set_title("Gray")
axes[2].imshow(binary_local, cmap="gray")
axes[2].set_title(f"Local Threshold Result\nBlock Size: {block_size} | Offset: {offset}")
plt.show()
