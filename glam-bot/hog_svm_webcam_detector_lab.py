import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.svm import SVC
import time


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


# Preprocessing function
def preprocess_image(img, method="clahe"):
    try:
        if method == "clahe":
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR + COLOR_BGR2Lab)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            img_lab = cv2.merge((l_clahe, a, b))
            img = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
        elif method == "unsharp":
            gaussian = cv2.GaussianBlur(img, (9, 9), 2.0)
            img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
        elif method == "bilateral":
            img = cv2.bilateralFilter(img, 9, 75, 75)
        elif method == "combined":
            # CLAHE
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            l, a, b = cv2.split(img_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_clahe = clahe.apply(l)
            img_lab = cv2.merge((l_clahe, a, b))
            img = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)
            # Unsharp Masking
            gaussian = cv2.GaussianBlur(img, (9, 9), 2.0)
            img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
            # Bilateral Filter
            img = cv2.bilateralFilter(img, 9, 75, 75)
        return img
    except Exception as e:
        print(f"Error in preprocessing ({method}): {e}")
        return img


# HOG+SVM Detector
def hog_svm_detector(frame, hog, bg_subtractor, preprocess_method):
    try:
        frame_gray = cv2.cvtColor(
            preprocess_image(frame, preprocess_method), cv2.COLOR_BGR2GRAY
        )
        fg_mask = bg_subtractor.apply(frame_gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        human_candidates = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 50 and h > 100 and 1.5 < (h / w) < 3.0:
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(frame_gray.shape[1] - x, w + 2 * pad)
                h = min(frame_gray.shape[0] - y, h + 2 * pad)
                if w >= 64 and h >= 128:
                    human_candidates.append((x, y, w, h))

        boxes = []
        scores = []
        for x, y, w, h in human_candidates:
            roi = frame_gray[y : y + h, x : x + w]
            if roi.size == 0 or roi.shape[0] < 128 or roi.shape[1] < 64:
                continue
            try:
                detections, weights = hog.detectMultiScale(
                    roi, winStride=(8, 8), padding=(8, 8), scale=1.05
                )
                for (hx, hy, hw, hh), score in zip(detections, weights):
                    if score > 0.6:
                        boxes.append([x + hx, y + hy, x + hx + hw, y + hy + hh])
                        scores.append(score)
            except Exception as e:
                print(f"Error in HOG+SVM detection: {e}")
                continue

        if boxes:
            boxes, scores = non_max_suppression(boxes, scores, overlapThresh=0.5)
        return boxes, scores
    except Exception as e:
        print(f"Error in HOG+SVM detector: {e}")
        return [], []


# HOG+LBP+SVM Detector (Simplified with heuristic threshold)
def hog_lbp_svm_detector(frame, hog, bg_subtractor, preprocess_method):
    try:
        frame_gray = cv2.cvtColor(
            preprocess_image(frame, preprocess_method), cv2.COLOR_BGR2GRAY
        )
        fg_mask = bg_subtractor.apply(frame_gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        human_candidates = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 50 and h > 100 and 1.5 < (h / w) < 3.0:
                pad = 10
                x = max(0, x - pad)
                y = max(0, y - pad)
                w = min(frame_gray.shape[1] - x, w + 2 * pad)
                h = min(frame_gray.shape[0] - y, h + 2 * pad)
                if w >= 64 and h >= 128:
                    human_candidates.append((x, y, w, h))

        boxes = []
        scores = []
        for x, y, w, h in human_candidates:
            roi = frame_gray[y : y + h, x : x + w]
            if roi.size == 0 or roi.shape[0] < 128 or roi.shape[1] < 64:
                continue
            try:
                # HOG features
                hog_features = hog.compute(roi).flatten()
                # LBP features
                lbp = local_binary_pattern(roi, P=8, R=1, method="uniform")
                lbp_hist, _ = np.histogram(lbp, bins=256, density=True)
                # Simple heuristic: combine HOG and LBP with threshold
                hog_score = np.mean(hog_features)
                lbp_score = np.max(lbp_hist)
                combined_score = 0.7 * hog_score + 0.3 * lbp_score
                if combined_score > 0.5:  # Arbitrary threshold
                    boxes.append([x, y, x + w, y + h])
                    scores.append(combined_score)
            except Exception as e:
                print(f"Error in HOG+LBP+SVM detection: {e}")
                continue

        if boxes:
            boxes, scores = non_max_suppression(boxes, scores, overlapThresh=0.5)
        return boxes, scores
    except Exception as e:
        print(f"Error in HOG+LBP+SVM detector: {e}")
        return [], []


# Cascade Classifier Detector
def cascade_detector(frame, cascade, preprocess_method):
    try:
        frame_gray = cv2.cvtColor(
            preprocess_image(frame, preprocess_method), cv2.COLOR_BGR2GRAY
        )
        detections = cascade.detectMultiScale(
            frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 100)
        )
        boxes = [[x, y, x + w, y + h] for (x, y, w, h) in detections]
        scores = [1.0] * len(boxes)  # No confidence scores, use 1.0
        if boxes:
            boxes, scores = non_max_suppression(boxes, scores, overlapThresh=0.5)
        return boxes, scores
    except Exception as e:
        print(f"Error in Cascade detector: {e}")
        return [], []


# Main function
def main():
    # Initialize detectors
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception as e:
        print(f"Error initializing HOG: {e}")
        exit()

    # try:
    #     cascade = cv2.CascadeClassifier(
    #         cv2.data.haarcascades + "haarcascade_fullbody.xml"
    #     )
    #     if cascade.empty():
    #         print("Error: Could not load Haar cascade.")
    #         exit()
    except Exception as e:
        print(f"Error initializing Cascade: {e}")
        exit()

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=16, detectShadows=True
    )

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    preprocess_methods = ["clahe", "unsharp", "bilateral", "combined"]
    detectors = [
        ("HOG+SVM", hog_svm_detector),
        ("HOG+LBP+SVM", hog_lbp_svm_detector),
    ]

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break
            frame_count += 1
            print(f"\nProcessing frame {frame_count}")

            # Resize for faster processing
            frame = cv2.resize(frame, (640, 480))

            # Compare detectors with different preprocessing
            for preprocess_method in preprocess_methods:
                print(f"\nPreprocessing: {preprocess_method}")
                frame_processed = frame.copy()

                for detector_name, detector_func in detectors:
                    start_det_time = time.time()
                    boxes, scores = detector_func(
                        frame_processed,
                        hog,
                        bg_subtractor,
                        preprocess_method,
                    )
                    det_time = time.time() - start_det_time

                    # Draw detections
                    for (x1, y1, x2, y2), score in zip(boxes, scores):
                        if (
                            x1 >= 0
                            and y1 >= 0
                            and x2 <= frame.shape[1]
                            and y2 <= frame.shape[0]
                        ):
                            cv2.rectangle(
                                frame_processed, (x1, y1), (x2, y2), (0, 255, 0), 2
                            )
                            cv2.putText(
                                frame_processed,
                                f"{detector_name}: {score:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                2,
                            )

                    print(
                        f"{detector_name}: {len(boxes)} detections, Time: {det_time:.3f}s"
                    )

                # Display result
                cv2.imshow(f"Detections ({preprocess_method})", frame_processed)

            if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        fps = frame_count / (time.time() - start_time)
        print(f"Average FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
