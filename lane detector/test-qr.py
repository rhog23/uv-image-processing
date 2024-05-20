import pyqrcode, cv2

# qr_generated = pyqrcode.create("1")
# qr_generated.png("qr.png", scale=5)

detector = cv2.QRCodeDetector()

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    data, vertices_array, binary_qrcode = detector.detectAndDecode(frame)

    if vertices_array is not None:
        print(data)

    cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


# image = cv2.imread("qr.png")
