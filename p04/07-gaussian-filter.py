import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../images/1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

result = cv2.GaussianBlur(img, (3, 3), 0)

fig, axs = plt.subplots(ncols=2)
axs[0].set_title("Original")
axs[0].set_axis_off()
axs[0].imshow(img)
axs[1].set_title("Result 1")
axs[1].set_axis_off()
axs[1].imshow(result)

plt.show()