import cv2
import numpy as np

img = cv2.imread("smaller-image-02.jpg")
h, w = img.shape[:2]

# Translasi ke kanan 50px dan ke bawah 30px
M = np.float32([[1, 0, 500], [0, 1, 30]])
translated = cv2.warpAffine(img, M, (w, h))

cv2.imshow("Translated Image", translated)
cv2.waitKey(0)
cv2.destroyAllWindows()
