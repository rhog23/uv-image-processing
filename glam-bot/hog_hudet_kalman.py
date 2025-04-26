import cv2
import numpy as np


def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
    )
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    return kalman


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Initialize HOG person detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Initialize Kalman filter
    kalman = initialize_kalman()

    # Start
    print("Press ESC to exit...")

    # First frame, assume no measurement yet
    has_measurement = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # Detect humans
        boxes, weights = hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )

        # If a detection is available
        if len(boxes) > 0:
            # Pick the largest detected person
            areas = [w * h for (x, y, w, h) in boxes]
            biggest_idx = np.argmax(areas)
            x, y, w, h = boxes[biggest_idx]

            # Center of detection
            center_x = x + w // 2
            center_y = y + h // 2

            measurement = np.array([[np.float32(center_x)], [np.float32(center_y)]])

            if not has_measurement:
                # Initialize Kalman state at first measurement
                kalman.statePre = np.array(
                    [[center_x], [center_y], [0], [0]], np.float32
                )
                has_measurement = True

            kalman.correct(measurement)

            # Draw detection box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

        # Predict next position
        prediction = kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        # Draw predicted position
        cv2.circle(frame, (pred_x, pred_y), 8, (0, 0, 255), -1)
        cv2.putText(
            frame,
            "Predicted",
            (pred_x + 10, pred_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Human Tracking with Kalman Filter", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
