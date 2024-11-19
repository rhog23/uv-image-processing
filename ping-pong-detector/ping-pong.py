import cv2
from ultralytics import YOLO

# model = YOLO("./models/ping-pong-det_saved_model/ping-pong-det_int8.tflite", task="detect")
model = YOLO("./models/pingpong-det-small_openvino_model", task="detect")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    _, frame = cap.read()

    results = model(frame, imgsz=160, conf=0.6)

    for result in results:
        annotated_image = result.plot()

    cv2.imshow("result", annotated_image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
