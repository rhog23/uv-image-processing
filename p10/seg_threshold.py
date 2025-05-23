import cv2
from skimage import filters
import numpy as np

# Membaca gambar dalam format grayscale
image = cv2.imread("images/WIN_20250516_20_13_50_Pro.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, None, fx=0.5, fy=0.5)

# Menerapkan global thresholding dengan nilai threshold 127
_, binary_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Menerapkan Otsu's thresholding (menentukan threshold secara otomatis)
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

# Menerapkan Li's thresholding
thresh_li = filters.threshold_li(image.astype(np.uint8))
binary_li = image > thresh_li  # matriks dengan format boolean
binary_li = binary_li.astype(np.uint8) * 255  # mengubah format boolean menjadi uint

# Menerapkan Yen's thresholding
thresh_yen = filters.threshold_yen(image.astype(np.uint8))
binary_yen = image > thresh_yen  # matriks dengan format boolean
binary_yen = binary_yen.astype(np.uint8) * 255  # mengubah format boolean menjadi uint

# Menerapkan Triangle thresholding
thresh_tri = filters.threshold_triangle(image.astype(np.uint8))
binary_tri = image > thresh_tri  # matriks dengan format boolean
binary_tri = binary_tri.astype(np.uint8) * 255  # mengubah format boolean menjadi uint


# Menampilkan hasil
cv2.imshow("Original Image", image)
cv2.imshow("Global Thresholding (T=127)", binary_thresh)
cv2.imshow("Otsu's Thresholding", otsu_thresh)
cv2.imshow("Li's Thresholding", binary_li)
cv2.imshow("Yen's Thresholding", binary_yen)
cv2.imshow("Triangle Thresholding", binary_tri)

cv2.waitKey(0)
cv2.destroyAllWindows()
