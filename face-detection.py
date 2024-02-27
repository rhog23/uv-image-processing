import cv2, time
from centerface import CenterFace

# Face Detection
def detect_faces(img):
    img = cv2.convertScaleAbs(img, alpha=2.5, beta=0)
    tinggi, lebar = img.shape[:2]
    centerface = CenterFace(landmarks=True)
  
    dets, lms = centerface(img, tinggi, lebar, threshold=0.35)

    for det in dets:
        bbox = det[:4]
        y = int(bbox[1])
        h = int(bbox[3])
        x = int(bbox[0])
        w = int(bbox[2])

        cv2.rectangle(img, (x, y), (w, h), (0,0,0), 2)
    
    return img


cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    center_x = width // 2
    center_y = height // 2

    # calculating the roi-box coordinates
    top_left_x = center_x - 60
    top_left_y = center_y - 60
    bottom_right_x = top_left_x + 120
    bottom_right_y = top_left_y + 120

    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)
    bottom_right_x = min(width, bottom_right_x)
    bottom_right_y = min(height, bottom_right_y)

    frame = cv2.convertScaleAbs(frame, alpha=0.25, beta=0)

    # draw roi box
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (70, 57, 230), 2)

    rect_img = frame[
        top_left_y : bottom_right_y, top_left_x : bottom_right_x
    ]

    frame[
        top_left_y : bottom_right_y, top_left_x : bottom_right_x
    ] = detect_faces(rect_img)
    lokasi_wajah = detect_faces(rect_img)

    new_frame_time = time.time()
    fps = f"[FPS]: {str(int(1 / (new_frame_time - prev_frame_time)))}"
    prev_frame_time = new_frame_time

    cv2.putText(
        frame,
        fps,
        (0, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (236, 56, 131),
        1,
        cv2.LINE_AA,
    )
    
    cv2.imshow("face detection demo", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()