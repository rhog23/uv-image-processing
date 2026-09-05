from ultralytics import YOLO
import cv2

model = YOLO(model="yolo12n.pt", task="detect")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()

    results = model(frame)

    frame_annot = results[0].plot()

    cv2.imshow("Hasil YOLO", frame_annot)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
