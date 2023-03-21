import cv2
import numpy as np
from skimage.feature import match_template
import matplotlib.pyplot as plt

img = cv2.imread("images/bw_grid.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h_mask = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
v_mask = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
d_mask1 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
d_mask2 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])
result = match_template(img, h_mask)
plt.imshow(result, cmap="gray")
plt.show()
