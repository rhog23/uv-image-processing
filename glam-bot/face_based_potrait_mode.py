import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    print("Press ESC to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=8, minSize=(40, 40)
        )

        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

        for x, y, w, h in faces:
            cv2.rectangle(
                mask, (x - 30, y - 30), (x + w + 30, y + h + 30), (255), thickness=-1
            )

        mask = cv2.dilate(mask, np.ones((30, 30), np.uint8), iterations=2)
        mask_inv = cv2.bitwise_not(mask)

        blurred = cv2.GaussianBlur(frame, (35, 35), 0)

        fg = cv2.bitwise_and(frame, frame, mask=mask)
        bg = cv2.bitwise_and(blurred, blurred, mask=mask_inv)

        portrait = cv2.add(fg, bg)

        cv2.imshow("Face-Based Portrait Mode", portrait)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
