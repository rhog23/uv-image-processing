import cv2

# Baca gambar
img = cv2.imread("smaller-image-02.jpg", cv2.IMREAD_GRAYSCALE)

# Membalik citra
img_inverted = cv2.bitwise_not(img)

# Tampilkan gambar asli dan gambar terbalik
cv2.imshow("Original Image", img)
cv2.imshow("Inverted Image", img_inverted)
cv2.waitKey(0)
cv2.destroyAllWindows()
