import cv2
import numpy as np
import time


def get_skin_mask(frame):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    cr_min, cr_max = 133, 173
    cb_min, cb_max = 77, 127
    skin_mask = cv2.inRange(ycrcb, (0, cr_min, cb_min), (255, cr_max, cb_max))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return skin_mask


def draw_hand_contours(frame, skin_mask):
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)


def draw_bounding_boxes(frame, skin_mask):
    contours, _ = cv2.findContours(
        skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def hand_gesture_count(frame, skin_mask):
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
                    a = np.linalg.norm(np.array(end) - np.array(start))
                    b = np.linalg.norm(np.array(far) - np.array(start))
                    c = np.linalg.norm(np.array(far) - np.array(end))
                    angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 180 / np.pi
                    if angle <= 90:
                        count += 1
                        cv2.circle(frame, far, 5, (0, 0, 255), -1)
                fingers = count + 1
                cv2.putText(
                    frame,
                    f"Fingers: {fingers}",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (255, 0, 0),
                    3,
                )


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    mode = 1  # Start with mode 1

    print("Press 1: Hand Tracking (Contours)")
    print("Press 2: Skin Bounding Boxes")
    print("Press 3: Hand Gesture Counting")
    print("Press ESC: Exit")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        skin_mask = get_skin_mask(frame)

        if mode == 1:
            draw_hand_contours(frame, skin_mask)
            cv2.putText(
                frame,
                "Mode 1: Hand Tracking",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        elif mode == 2:
            draw_bounding_boxes(frame, skin_mask)
            cv2.putText(
                frame,
                "Mode 2: Bounding Boxes",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
        elif mode == 3:
            hand_gesture_count(frame, skin_mask)
            cv2.putText(
                frame,
                "Mode 3: Gesture Counting",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (500, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Skin Detection Multi-Mode with FPS", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord("1"):
            mode = 1
        elif key == ord("2"):
            mode = 2
        elif key == ord("3"):
            mode = 3

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
