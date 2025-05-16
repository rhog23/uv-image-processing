import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Load the cat face cascade classifier
    cat_face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt.xml"
    )

    if cat_face_cascade.empty():
        print("Error loading cat face cascade")
        return

    print("Press ESC to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect cat faces
        cat_faces = cat_face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in cat_faces:
            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),  # Green color
                2,  # Thickness
            )

            # Calculate centroid (center point)
            centroid_x = x + w // 2
            centroid_y = y + h // 2

            # Draw centroid point
            cv2.circle(
                frame,
                (centroid_x, centroid_y),
                5,  # Radius
                (0, 0, 255),  # Red color
                -1,  # Filled circle
            )

            # Display coordinates text
            cv2.putText(
                frame,
                f"({centroid_x}, {centroid_y})",
                (centroid_x + 10, centroid_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Cat Face Detection with Centroid", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
