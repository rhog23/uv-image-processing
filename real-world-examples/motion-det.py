import cv2
import numpy as np


def main():
    # Initialize the video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create the MOG2 background subtractor
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=0, varThreshold=60, detectShadows=True
    )

    print("Press 'q' to exit.")

    # Set up the window for fullscreen display
    window_name = "Fullscreen Quad View"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Resize frame for consistency (optional)
        frame_resized = cv2.resize(frame, (640, 480))

        # Apply the MOG2 background subtractor
        fg_mask = mog2.apply(frame_resized)

        # Optionally remove noise from the foreground mask
        fg_mask = cv2.medianBlur(fg_mask, 5)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw bounding boxes around detected motions and label the areas
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small areas
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame_resized,
                    f"Area: {cv2.contourArea(contour)}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Background model visualization
        background = mog2.getBackgroundImage()
        if background is not None:
            background_resized = cv2.resize(background, (640, 480))
        else:
            background_resized = np.zeros_like(frame_resized)

        # Frame difference visualization (difference between current frame and background model)
        if background is not None:
            frame_diff = cv2.absdiff(background, frame_resized)
        else:
            frame_diff = np.zeros_like(frame_resized)

        # Convert fg_mask to a 3-channel image for consistency
        fg_mask_colored = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)

        # Stack the four images into a single image with 4 quadrants
        top_row = np.hstack((frame_resized, fg_mask_colored))  # Concatenate first row
        bottom_row = np.hstack(
            (background_resized, frame_diff)
        )  # Concatenate second row

        # Concatenate the two rows vertically
        full_screen_frame = np.vstack((top_row, bottom_row))

        # Display the final fullscreen frame
        cv2.imshow(window_name, full_screen_frame)

        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
