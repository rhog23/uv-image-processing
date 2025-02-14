import cv2
import numpy as np

img = cv2.imread("smaller-image-02.jpg")

kernel = np.ones((5, 5), np.uint8)
result = cv2.dilate(img, kernel)

# Menampilkan hasil
cv2.imshow("Original", img)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
