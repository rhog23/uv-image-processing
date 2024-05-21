import cv2
from ultralytics import YOLO

model = YOLO("models/ping-pong-det_saved_model/ping-pong-det_int8.tflite", task="detect")
# clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    _, frame = cap.read()
    # ycrcb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # ycrcb_img[:, :, 0] = clahe.apply(ycrcb_img[:, :, 0])

    # equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

    results = model(frame, imgsz=160, conf=0.6)

    for result in results:
        annotated_image = result.plot()

    cv2.imshow("result", annotated_image)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
