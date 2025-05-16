import cv2
import numpy as np

# Load LBP cascade for face detection
lbp_face_cascade = cv2.CascadeClassifier()
lbp_face_cascade.load(cv2.samples.findFile("./data/lbpcascade_frontalface.xml"))


def lbp_detection():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # LBP face detection
        faces = lbp_face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        # Draw detections
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("LBP Face Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Run LBP detector
lbp_detection()
