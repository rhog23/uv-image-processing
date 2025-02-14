import cv2

# Membaca gambar dalam mode grayscale
img = cv2.imread("tsukuba_l.png", cv2.IMREAD_GRAYSCALE)

clahe = cv2.createCLAHE(clipLimit=2)
clahe_img = clahe.apply(img)

# Menampilkan hasil
cv2.imshow("Original Grayscale Image", img)
cv2.imshow("CLAHE", clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
