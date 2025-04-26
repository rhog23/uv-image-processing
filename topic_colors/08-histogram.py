import cv2
import matplotlib.pyplot as plt

img = cv2.imread("../images/1.png", cv2.IMREAD_GRAYSCALE)
# plt.subplot(121)
# plt.imshow(img, cmap="gray")
# plt.subplot(122)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.show()

# cara 2
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.subplot(121)
plt.imshow(img, cmap="gray")
plt.subplot(122)
plt.plot(hist)
plt.show()

# rgb
img = cv2.imread("../images/1.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
color = ("r", "g", "b")
plt.subplot(121)
plt.imshow(img_rgb)
plt.subplot(122)
for i, col in enumerate(color):
    hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
plt.show()
