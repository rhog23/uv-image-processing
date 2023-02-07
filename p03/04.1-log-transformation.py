import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("../images/1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

c = 255 / np.log(1 + np.max(img))

img_log = np.array(c * np.log(1 + img), dtype="uint8")

fig, axs = plt.subplots(ncols=2)
axs[0].set_title("Original Image")
axs[0].set_axis_off()
axs[0].imshow(img)
axs[1].set_title("Result")
axs[1].set_axis_off()
axs[1].imshow(img_log)
plt.show()
