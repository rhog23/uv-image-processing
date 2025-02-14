import cv2

img = cv2.imread("smaller-image-02.jpg")

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = cv2.blur(img, (3, 3))

# Menampilkan hasil
cv2.imshow("Original", img)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
