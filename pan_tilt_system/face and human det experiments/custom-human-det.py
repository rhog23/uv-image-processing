# source: https://github.com/RashadGarayev/PersonDetection
import numpy as np
from skimage.transform import pyramid_gaussian
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
import joblib, cv2
from skimage import color
import utils.sliding as sd

# Parameters
size = (64, 128)  # HOG window size
step_size = (10, 10)  # sliding window step size
downscale = 1.25  # pyramid downscale factor
min_confidence = 0.5  # minimum confidence threshold

# Load the trained model
try:
    model = joblib.load("data/models.dat")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize video capture
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Resize frame for consistent processing
    image = cv2.resize(frame, (512, 512))
    detections = []
    scale = 0

    # Image pyramid
    for im_scaled in pyramid_gaussian(image, downscale=downscale):
        # Stop if image is smaller than detection window
        if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
            break

        # Convert to grayscale for HOG
        gray = color.rgb2gray(im_scaled)

        # Sliding window
        for x, y, window in sd.sliding_window(im_scaled, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue

            # Get HOG features
            fd = hog(
                color.rgb2gray(window),
                orientations=9,
                pixels_per_cell=(8, 8),
                visualize=False,
                cells_per_block=(3, 3),
            ).reshape(1, -1)

            # Make prediction
            pred = model.predict(fd)
            confidence = model.decision_function(fd)[0]

            if pred == 1 and confidence > min_confidence:
                # Scale the coordinates back to original image size
                x_scaled = int(x * (downscale**scale))
                y_scaled = int(y * (downscale**scale))
                w_scaled = int(size[0] * (downscale**scale))
                h_scaled = int(size[1] * (downscale**scale))

                detections.append((x_scaled, y_scaled, confidence, w_scaled, h_scaled))

        scale += 1

    # Process detections
    clone = image.copy()
    if len(detections) > 0:
        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        scores = np.array([score for (_, _, score, _, _) in detections])

        # Apply non-maxima suppression
        pick = non_max_suppression(rects, probs=scores, overlapThresh=0.3)

        # Draw bounding boxes
        for x1, y1, x2, y2 in pick:
            cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                clone,
                f"Person: {scores.max():.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    # Display output
    cv2.imshow("Person Detection", clone)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
