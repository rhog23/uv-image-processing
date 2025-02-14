import cv2

# Membaca gambar berwarna
img = cv2.imread("smaller-image-02.jpg")

clahe = cv2.createCLAHE()

# Konversi gambar ke ruang warna YCrCb (Luminance + Chrominance)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# Pisahkan channel Y (luminance), Cr, dan Cb
Y, Cr, Cb = cv2.split(ycrcb)

# Lakukan Histogram Equalization hanya pada channel Y
Y_eq = cv2.equalizeHist(Y)

# Gabungkan kembali dengan channel Cr dan Cb yang asli
ycrcb_eq = cv2.merge([Y_eq, Cr, Cb])

# Konversi kembali ke BGR untuk ditampilkan
equalized_img = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

# Menampilkan hasil
cv2.imshow("Original Image", img)
cv2.imshow("Histogram Equalized Image (RGB)", equalized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
