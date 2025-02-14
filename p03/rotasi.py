import cv2

img = cv2.imread("smaller-image-02.jpg")
h, w = img.shape[:2]

# Rotasi 45 derajat terhadap pusat gambar
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1)
rotated = cv2.warpAffine(img, M, (w, h))

cv2.imshow("Rotated Image", rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
