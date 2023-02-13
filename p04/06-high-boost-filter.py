import numpy as np
import matplotlib.pyplot as plt
import cv2

# Defining kernels
k_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
k_2 = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]], dtype=np.float32)
k_3 = np.array([[-1, -1, -1], [-1, 11, -1], [-1, -1, -1]], dtype=np.float32)

img = cv2.imread("../images/1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r_1 = cv2.filter2D(img, -1, k_1)
r_2 = cv2.filter2D(img, -1, k_2)
r_3 = cv2.filter2D(img, -1, k_3)

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0, 0].set_title("Original")
axs[0, 0].set_axis_off()
axs[0, 0].imshow(img)
axs[0, 1].set_title("Result 1")
axs[0, 1].set_axis_off()
axs[0, 1].imshow(r_1)
axs[1, 0].set_title("Result 2")
axs[1, 0].set_axis_off()
axs[1, 0].imshow(r_2)
axs[1, 1].set_title("Result 3")
axs[1, 1].set_axis_off()
axs[1, 1].imshow(r_3)
plt.show()
