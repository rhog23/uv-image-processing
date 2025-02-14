import cv2

# Inisialisasi detektor HOG + SVM bawaan OpenCV
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Buka webcam (0 untuk kamera utama)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
    # Konversi frame ke grayscale (opsional, bisa tetap warna)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi manusia dalam frame
    bodies, _ = hog.detectMultiScale(gray, winStride=(4, 4), padding=(4, 4), scale=1.1)

    # Gambar kotak di sekitar manusia yang terdeteksi
    for x, y, w, h in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Tampilkan hasil deteksi
    cv2.imshow("HOG Human Detection", frame)

    # Keluar jika menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
