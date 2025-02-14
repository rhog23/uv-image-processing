import cv2

# Membaca citra
img = cv2.imread("smaller-image-02.jpg")

# Konversi ke YCrCb
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Pisahkan channel Y, Cr, Cb
y, cr, cb = cv2.split(ycrcb)

clahe = cv2.createCLAHE(clipLimit=2)

# Terapkan CLAHE hanya pada channel Y
y_eq = clahe.apply(y)

# Gabungkan kembali channel Y yang telah diolah
ycrcb_eq = cv2.merge([y_eq, cr, cb])

# Konversi kembali ke RGB
img_clahe = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# Tampilkan hasil
cv2.imshow("Original Image", img)
cv2.imshow("CLAHE Enhanced Image", img_clahe)

cv2.waitKey(0)
cv2.destroyAllWindows()
