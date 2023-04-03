import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import io
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk
from skimage.color import label2rgb

image = io.imread("images/jahe.jpg", as_gray=True)

# apply threshold
thresh = threshold_otsu(image)
bw = closing(image > thresh, disk(3))
bw = np.bitwise_not(bw)
cleared = clear_border(bw)
label_image = label(cleared)
image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
for region in regionprops(label_image):
    print("Area: ", region.area)
    print("Perimeter: ", region.perimeter)

fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(image_label_overlay)


ax.set_axis_off()
plt.tight_layout()
plt.show()
