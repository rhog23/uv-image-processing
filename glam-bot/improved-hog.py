import cv2
import numpy as np


# Non-Maximum Suppression to merge overlapping detections
def non_max_suppression(boxes, scores, overlapThresh=0.5):
    if len(boxes) == 0:
        return [], []

    boxes = np.array(boxes, dtype="float")
    scores = np.array(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlapThresh)[0]
        order = order[inds + 1]

    return boxes[keep].astype("int"), scores[keep]


# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=16, detectShadows=True
)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame_count += 1
        print(f"Processing frame {frame_count}")

        # Resize for faster processing
        frame = cv2.resize(frame, (640, 480))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        frame_gray = clahe.apply(frame_gray)

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame_gray)

        # Clean foreground mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in foreground mask
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Filter contours for human-like regions
        human_candidates = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 50 and h > 100 and 1.5 < (h / w) < 3.0:
                human_candidates.append((x, y, w, h))

        # Apply HOG+SVM to candidate regions
        all_boxes = []
        all_scores = []
        for x, y, w, h in human_candidates:
            # Validate ROI coordinates
            if (
                x < 0
                or y < 0
                or x + w > frame_gray.shape[1]
                or y + h > frame_gray.shape[0]
            ):
                print(f"Invalid ROI: ({x}, {y}, {w}, {h})")
                continue
            roi = frame_gray[y : y + h, x : x + w]
            if roi.size == 0:
                print(f"Empty ROI: ({x}, {y}, {w}, {h})")
                continue
            print(f"ROI shape: {roi.shape}")
            boxes, weights = hog.detectMultiScale(
                roi, winStride=(4, 4), padding=(8, 8), scale=1.02
            )
            if boxes is None or weights is None:
                print("Error: detectMultiScale returned None")
                continue
            for (hx, hy, hw, hh), score in zip(boxes, weights):
                if score > 0.6:
                    all_boxes.append([x + hx, y + hy, x + hx + hw, y + hy + hh])
                    all_scores.append(score)

        # Apply NMS
        if all_boxes:
            boxes, scores = non_max_suppression(
                all_boxes, all_scores, overlapThresh=0.5
            )
        else:
            boxes, scores = [], []

        # Draw detections
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            # Validate drawing coordinates
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                print(f"Out-of-bound detection: ({x1}, {y1}, {x2}, {y2})")
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Display result
        cv2.imshow("HOG Body Detection", frame)
        if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
