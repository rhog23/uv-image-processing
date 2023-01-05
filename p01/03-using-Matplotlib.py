import matplotlib.image as mpimg
import matplotlib.pyplot as plt

im = mpimg.imread("../images/1.png")
im1 = im.copy()
im1[im1 < 0.5] = 0
print(f"{'Image Shape':<20}:{im.shape}")
print(f"{'Image Data Type':<20}:{im.dtype}")
print(f"{'Type':<20}:{type(im)}")
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
ax1.imshow(im)
ax2.imshow(im1)
ax1.axis("off")
ax2.axis("off")
plt.show()
