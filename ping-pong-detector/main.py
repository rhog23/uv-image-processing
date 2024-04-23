import cv2
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("models/pingpong-det-small_openvino_model", task="detect")

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        results = model(frame, imgsz=160)

        for result in results:
            for box in result.boxes:
                coords = box.xyxy

                if len(coords) > 0:
                    x = int(coords[0][0])
                    y = int(coords[0][1])
                    w = int(coords[0][2])
                    h = int(coords[0][3])

                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 2)
                else:
                    continue
                

            cv2.imshow("detection result", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
