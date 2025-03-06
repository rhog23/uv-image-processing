import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

# Membaca gambar dalam format grayscale
image = cv2.imread("images/WIN_20230502_21_35_07_Pro.jpg", cv2.IMREAD_GRAYSCALE)

# Menerapkan global thresholding dengan nilai threshold 127 (binary threshold)
ret_binary, binary_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Menerapkan Otsu's thresholding (nilai threshold dihitung secara otomatis)
ret_otsu, otsu_thresh = cv2.threshold(
    image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Menerapkan Li's tresholding (nilai threshold dihitung secara otomatis)
thresh_li = filters.threshold_li(image.astype(np.uint8))

# Menerapkan Yen's tresholding (nilai threshold dihitung secara otomatis)
thresh_yen = filters.threshold_yen(image.astype(np.uint8))

# Menerapkan Triangle tresholding (nilai threshold dihitung secara otomatis)
thresh_tri = filters.threshold_triangle(image.astype(np.uint8))

# Menghitung histogram citra (256 bins untuk intensitas 0-255)
hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# Menampilkan histogram menggunakan matplotlib
plt.figure(figsize=(10, 5))
plt.plot(hist, color="gray", label="Histogram")

# Menggambar garis merah untuk binary threshold (T = 127)
plt.axvline(x=127, color="red", linestyle="--", label="Binary Threshold = 127")

# Menggambar garis biru untuk Otsu's threshold (nilai yang dihitung otomatis)
plt.axvline(
    x=ret_otsu, color="blue", linestyle="--", label=f"Otsu's Threshold = {ret_otsu:.2f}"
)

# Menggambar garis hijau untuk Li's threshold (nilai yang dihitung otomatis)
plt.axvline(
    x=thresh_li,
    color="green",
    linestyle="--",
    label=f"Li's Threshold = {thresh_li:.2f}",
)

# Menggambar garis hijau untuk Yen's threshold (nilai yang dihitung otomatis)
plt.axvline(
    x=thresh_yen,
    color="orange",
    linestyle="--",
    label=f"Yen's Threshold = {thresh_yen:.2f}",
)

# Menggambar garis hijau untuk Triangle threshold (nilai yang dihitung otomatis)
plt.axvline(
    x=thresh_tri,
    color="purple",
    linestyle="--",
    label=f"Triangle's Threshold = {thresh_tri:.2f}",
)

plt.xlabel("Intensitas Piksel")
plt.ylabel("Frekuensi")
plt.title("Histogram Citra serta Nilai Threshold (Ambang) untuk Segmentasi")
plt.legend()
plt.show()
