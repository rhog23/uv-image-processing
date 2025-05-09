from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np

box_annotator = sv.BoundingBoxAnnotator()
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)
# model = YOLO("yolo11n-seg.pt", task="segment")
# model = YOLO("yolo11n_openvino_model", task="detect")
model = YOLO("fruit-det-y12-nadam.pt", task="detect")

cap = cv2.VideoCapture(0)


def draw_bbox_with_centroid(frame, detections):
    """
    Draw bounding boxes and their centroids on the frame
    Returns the modified frame
    """
    # Create a copy of the frame to draw on
    annotated_frame = frame.copy()

    # Get bounding box coordinates (xyxy format)
    if len(detections.xyxy) > 0:
        for i, bbox in enumerate(detections.xyxy):
            # Extract coordinates
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Draw the bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate centroid
            centroid_x = int((x1 + x2) / 2)
            centroid_y = int((y1 + y2) / 2)

            # Draw centroid (circle)
            cv2.circle(annotated_frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)

            # Get class name for this detection
            class_id = detections.class_id[i]
            class_name = results.names[class_id]

            # Display centroid coordinates and class name
            text = f"{class_name}: ({centroid_x}, {centroid_y})"
            cv2.putText(
                annotated_frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

    return annotated_frame


while True:
    success, frame = cap.read()

    if success:
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        labels = [f"{results.names[class_id]}" for class_id in detections.class_id]

        # Use custom function to draw bounding boxes and centroids
        annotated_frame = draw_bbox_with_centroid(frame, detections)

        # Add labels using supervision's label annotator
        annotated_frame = label_annotator.annotate(
            annotated_frame, detections=detections, labels=labels
        )

        # Display the frame
        cv2.imshow("Object Detection with Centroids", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
