import cv2
import numpy as np
import threading
import time

# Load classifiers
# 1. Haar frontal face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 2. Full body detection
full_body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

# Global variables for thread communication
latest_frame = None
processed_frame = None
lock = threading.Lock()
running = True


def detection_worker():
    global latest_frame, processed_frame, running

    while running:
        start_time = time.time()

        # Get frame with lock
        with lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downscale for faster processing (60% of original)
        scale_percent = 60
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        small_frame = cv2.resize(frame, (width, height))
        small_gray = cv2.resize(gray, (width, height))

        # Detect objects with optimized parameters
        faces = face_cascade.detectMultiScale(
            small_gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
        )

        full_bodies = full_body_cascade.detectMultiScale(
            small_gray, scaleFactor=1.05, minNeighbors=4, minSize=(60, 60)
        )

        # Scale detections back to original size
        scale_x = frame.shape[1] / width
        scale_y = frame.shape[0] / height

        # Draw detections on original frame
        for x, y, w, h in faces:
            cv2.rectangle(
                frame,
                (int(x * scale_x), int(y * scale_y)),
                (int((x + w) * scale_x), int((y + h) * scale_y)),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "Face",
                (int(x * scale_x), int(y * scale_y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        for x, y, w, h in full_bodies:
            cv2.rectangle(
                frame,
                (int(x * scale_x), int(y * scale_y)),
                (int((x + w) * scale_x), int((y + h) * scale_y)),
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                "Upper Body",
                (int(x * scale_x), int(y * scale_y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        # Calculate and display FPS
        fps = 1 / (time.time() - start_time)
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        # Update processed frame
        with lock:
            processed_frame = frame


def main():
    global latest_frame, running

    # Open webcam
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Start worker thread
    worker = threading.Thread(target=detection_worker)
    worker.daemon = True
    worker.start()

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update latest frame
        with lock:
            latest_frame = frame.copy()
            if processed_frame is not None:
                display_frame = processed_frame.copy()
            else:
                display_frame = frame.copy()

        # Show the result
        cv2.imshow("Face & Body Detection", display_frame)

        # Exit on ESC
        if cv2.waitKey(1) == 27:
            running = False
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    worker.join()


if __name__ == "__main__":
    main()
