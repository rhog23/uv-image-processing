import cv2
import numpy as np


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    print("Press ESC to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Many webcams deliver in YUV format internally; simulate converting:
        # Convert BGR -> YUV
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Extract channels
        y, u, v = cv2.split(frame_yuv)

        # Merge back for viewing (optionally adjust scaling)
        y_only = cv2.merge([y, y, y])  # Just brightness
        uv_only = cv2.merge([u, v, np.zeros_like(u)])  # UV components for fun

        # Show everything
        cv2.imshow("Original (BGR)", frame)
        cv2.imshow("Y Channel (Brightness)", y_only)
        cv2.imshow("U-V Components", uv_only)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
