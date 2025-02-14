import cv2

# Load Haar Cascade untuk deteksi manusia
human_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

# Buka video stream (0 untuk webcam utama)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Deteksi manusia
    humans = human_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(100, 100)
    )

    # Gambar kotak di sekitar manusia yang terdeteksi
    for x, y, w, h in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Tampilkan frame dengan deteksi
    cv2.imshow("Human Detection (Press 'q' to exit)", frame)

    # Keluar jika menekan 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Tutup kamera dan jendela OpenCV
cap.release()
cv2.destroyAllWindows()
