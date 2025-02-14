import cv2

# Membaca gambar dalam mode grayscale
img = cv2.imread("tsukuba_l.png", cv2.IMREAD_GRAYSCALE)

# Melakukan Histogram Equalization
equalized_img = cv2.equalizeHist(img)

# Menampilkan hasil
cv2.imshow("Original Grayscale Image", img)
cv2.imshow("Equalized Grayscale Image", equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
