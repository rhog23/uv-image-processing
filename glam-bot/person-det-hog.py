import cv2


def hog_body_detection():
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_gray = clahe.apply(frame_gray)

        # HOG detection
        bodies, _ = hog.detectMultiScale(
            frame_gray, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        # Draw detections
        for x, y, w, h in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("HOG Body Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Run HOG detector
hog_body_detection()
