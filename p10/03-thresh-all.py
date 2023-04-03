import matplotlib.pyplot as plt
from skimage import io, color, filters

img = io.imread("images/jahe.jpg")
assert img is not None, "file could not be read, please check the path"
gray = color.rgb2gray(img)
gray = filters.gaussian(gray, sigma=3)

fig, ax = filters.try_all_threshold(gray, figsize=(10, 8), verbose=False)
fig.suptitle("All Threshold", fontsize=20)
plt.show()
