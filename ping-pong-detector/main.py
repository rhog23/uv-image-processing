import cv2
from numpy import uint8
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("models/waste-detector_openvino_model", task="detect")

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        results = model(frame, conf=0.5)

        for result in results:
            for box in result.boxes:
                coords = box.xyxy

                if len(coords) > 0:
                    x = int(coords[0][0])
                    y = int(coords[0][1])
                    w = int(coords[0][2])
                    h = int(coords[0][3])

                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 255), 2)

                    cv2.putText(
                        frame,
                        f"{result.names[int(box.cls)]} | {box.conf[0]:.2%}",
                        (x, y),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    continue

            cv2.imshow("detection result", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
