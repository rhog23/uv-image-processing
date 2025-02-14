import cv2

img = cv2.imread("smaller-image-02.jpg")

result = cv2.GaussianBlur(img, (0, 0), 1)

# Menampilkan hasil
cv2.imshow("Original", img)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
