import cv2
import numpy as np


def stack_images(img_array, scale):
    rows = len(img_array)
    cols = len(img_array[0])
    heights = [img.shape[0] for img in img_array[0]]
    widths = [img.shape[1] for img in img_array[0]]

    max_height = max(heights)
    max_width = max(widths)

    # Resize all images to the same size
    for x in range(rows):
        for y in range(cols):
            img = img_array[x][y]
            if img.shape[:2] != (max_height, max_width):
                img_array[x][y] = cv2.resize(img, (max_width, max_height))

    hor = [np.hstack(img_array[x]) for x in range(rows)]
    stacked = np.vstack(hor)
    stacked = cv2.resize(stacked, (0, 0), fx=scale, fy=scale)
    return stacked


def create_histogram_image(channel, color):
    hist_img = np.zeros((300, 256, 3), dtype=np.uint8)
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    cv2.normalize(hist, hist, 0, hist_img.shape[0], cv2.NORM_MINMAX)
    hist = hist.flatten()

    for x in range(1, 256):
        cv2.line(
            hist_img,
            (x - 1, hist_img.shape[0] - int(hist[x - 1])),
            (x, hist_img.shape[0] - int(hist[x])),
            color,
            thickness=2,
        )
    return hist_img


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

        frame = cv2.resize(frame, (640, 480))

        # Convert BGR to YUV
        frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Split into Y, U, and V channels
        y_channel, u_channel, v_channel = cv2.split(frame_yuv)

        # Expand single channel to 3-channel grayscale images for visualization
        y_vis = cv2.merge([y_channel, y_channel, y_channel])
        u_vis = cv2.merge([u_channel, u_channel, u_channel])
        v_vis = cv2.merge([v_channel, v_channel, v_channel])

        # Create live histograms
        y_hist = create_histogram_image(y_channel, (255, 255, 255))  # White for Y
        u_hist = create_histogram_image(u_channel, (255, 0, 0))  # Blue for U
        v_hist = create_histogram_image(v_channel, (0, 0, 255))  # Red for V

        # Stack images into a nice 3x2 grid
        stacked_img = stack_images(
            [[frame, y_vis], [u_vis, v_vis], [y_hist, cv2.hconcat([u_hist, v_hist])]],
            scale=0.5,
        )

        cv2.imshow("Webcam YUV Channels + Live Histograms", stacked_img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
