import cv2
import numpy as np

# Load gambar
img = cv2.imread("white-gundam-thumb.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Tingkatkan kecerahan dengan menambahkan nilai ke setiap piksel
bright_img = cv2.convertScaleAbs(img, alpha=2.5, beta=0)  # beta meningkatkan kecerahan

# Tampilkan hasil
cv2.imshow("Original", img)
cv2.imshow("Brightness Enhanced", bright_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
