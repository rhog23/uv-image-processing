import serial
import time
from ultralytics import YOLO
import os
import cv2

# Initialize serial communication (replace with your port settings)
ser = serial.Serial("/dev/ttyUSB0", 9600)  # Adjust port name and baud rate

# Initialize YOLO model
model = YOLO("tomatoweight_int8_half_128.tflite", task="detect")

cap = cv2.VideoCapture(0)

# Define ROI size
roi_width, roi_height = 120, 120


def apply_clahe_rgb(image):
    # Split the image into R, G, and B channels
    r, g, b = cv2.split(image)

    # Create a CLAHE object
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))

    # Apply CLAHE to each channel
    r_clahe = clahe.apply(r)
    g_clahe = clahe.apply(g)
    b_clahe = clahe.apply(b)

    # Merge the channels back
    image_clahe = cv2.merge((r_clahe, g_clahe, b_clahe))

    return image_clahe


# Set frame processing interval (2 seconds in this example)
process_interval = 2  # Adjust in seconds

start_time = time.time()  # Initialize time for processing interval

while True:
    success, frame = cap.read()

    # Check if processing interval has elapsed
    if time.time() - start_time >= process_interval:
        # Resize frame to match the resolution in the second code snippet
        frame = cv2.resize(frame, (180, 170))

        frame_clahe = apply_clahe_rgb(frame)

        # Calculate ROI coordinates to center it in the frame
        frame_height, frame_width = frame_clahe.shape[:2]
        roi_x = (frame_width - roi_width) // 2
        roi_y = (frame_height - roi_height) // 2

        # Apply ROI
        roi = frame[roi_y : roi_y + roi_height, roi_x : roi_x + roi_width]

        results = model(roi, imgsz=128, int8=True, half=True, conf=0.90)

        tomato_detected = False
        unripe_detected = False
        not_tomato_detected = False
        no_detection = False

        for result in results:
            box = result.boxes
            coords = box.xyxy
            if len(coords) >= 1:
                x = int(coords[0][0])
                y = int(coords[0][1])
                w = int(coords[0][2])
                h = int(coords[0][3])

                cv2.rectangle(frame_clahe, (x, y), (w, h), (0, 255, 255), 1)
                cv2.putText(
                    frame_clahe,
                    f"{result.names[int(box.cls[0])]} | conf: {box.conf[0]:.2%}",
                    (x, y),
                    1,
                    0.7,
                    (0, 0, 255),
                    1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                class_name = result.names[0]  # Get the class name from YOLO output

                if class_name == "ripe":  # Check for specific class names
                    tomato_detected = True
                elif class_name == "unripe":
                    unripe_detected = True
                else:
                    not_tomato_detected = True
                break  # Stop iterating after finding something
            else:
                no_detection = True

        # Send serial command based on detection (within processing interval)
        if tomato_detected:
            ser.write("R".encode())  # Send 'R' for ripe tomato
            print("Ripe tomato detected, sending right servo command")
        elif unripe_detected:
            ser.write("U".encode())  # Send 'L' for unripe
            print("Unripe tomato detected, sending left servo command")
        else:
            ser.write("O".encode())  # Send 'O' for not-tomato
            print("Not a tomato detected, sending neutral servo command (optional)")

        # Reset start time for next processing interval
        start_time = time.time()

    # Rest of your code for drawing bounding boxes and ROI... (optional)
    cv2.rectangle(
        frame_clahe,
        (roi_x, roi_y),
        (roi_x + roi_width, roi_y + roi_height),
        (0, 255, 0),
        2,
    )
    cv2.imshow("detector", frame_clahe)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
ser.close()  # Close serial communication
cv2.destroyAllWindows()
