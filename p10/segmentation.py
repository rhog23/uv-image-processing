import cv2
from skimage import filters
import numpy as np

# Membaca gambar dalam format grayscale
image = cv2.imread("images/WIN_20230502_21_35_07_Pro.jpg")
image = cv2.resize(image, None, fx=0.5, fy=0.5)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Menerapkan Li's thresholding
thresh_li = filters.threshold_li(gray_image.astype(np.uint8))
binary_li = gray_image > thresh_li  # matriks dengan format boolean
binary_li = cv2.bitwise_not(
    binary_li.astype(np.uint8) * 255
)  # mengubah format boolean menjadi uint

# Menerapkan closing pada hasil citra biner
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# print(se)
result = cv2.morphologyEx(binary_li, cv2.MORPH_CLOSE, se)

# Memotong gambar asli dengan menggunakan mask yang telah diciptakan
masked_image = cv2.bitwise_and(src1=image, src2=image, mask=result)


# Menampilkan hasil
cv2.imshow("Original Image", image)
cv2.imshow("Li's Thresholding", binary_li)
cv2.imshow("Closing", result)
cv2.imshow("Final Result", masked_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
