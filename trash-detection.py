from ultralytics import YOLO
import supervision as sv
import cv2

box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
model = YOLO("ping-pong-detector/models/waste-detector_openvino_model", task="detect")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if success:
        results = model(frame)[0]

        detections = sv.Detections.from_ultralytics(results)

        labels = [f"{results.names[class_id]}" for class_id in detections.class_id]

        annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)

        cv2.imshow(
            "trash detector",
            label_annotator.annotate(
                annotated_frame, detections=detections, labels=labels
            ),
        )

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
