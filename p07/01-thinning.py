import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/bw-animals.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(thresh, bw) = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
thinned = cv2.ximgproc.thinning(src=bw)

plt.figure()
plt.imshow(thinned, cmap="gray")
plt.show()
