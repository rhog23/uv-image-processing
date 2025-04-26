import cv2
import numpy as np


def hand_gesture_counting():
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
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        cr_min, cr_max = 133, 173
        cb_min, cb_max = 77, 127

        skin_mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(cnt, returnPoints=False)

            if hull is not None and len(hull) > 3 and len(cnt) >= 5:
                defects = cv2.convexityDefects(cnt, hull)
                if defects is not None:
                    count = 0
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        # Filter by angle between fingers
                        a = np.linalg.norm(np.array(end) - np.array(start))
                        b = np.linalg.norm(np.array(far) - np.array(start))
                        c = np.linalg.norm(np.array(far) - np.array(end))
                        angle = (
                            np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 180 / np.pi
                        )
                        if angle <= 90:
                            count += 1
                            cv2.circle(frame, far, 5, (0, 0, 255), -1)
                    fingers = count + 1
                    cv2.putText(
                        frame,
                        f"Fingers: {fingers}",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (255, 0, 0),
                        3,
                    )

        cv2.imshow("Hand Gesture Counting", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# Uncomment to run
hand_gesture_counting()
