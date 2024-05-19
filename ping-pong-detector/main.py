import cv2
import time, sys, os
import picar, pathlib
from pymata4 import pymata4
from ultralytics import YOLO

model_path = "models/pingpong-tflite/pingpong-det-small_int8.tflite"

# Ultrasonic Sensor's Pin
trigger_pin = 8  #  digital input 8
echo_pin = 9  #  digital input 9

ena_pin = 11
enb_pin = 12

motor_speed = (
    150  #  changes the current to ena and enb pin resulting in changes of speed
)

# Servo's Pin
servo_pin = 10
distance = 100  # initial distance is set to 30 cm

# Wheel's Pin
left_motor_FW = 7
left_motor_BW = 6
right_motor_FW = 5
right_motor_BW = 4

ena = [ena_pin, enb_pin]
motor = [left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW]

board = pymata4.Pymata4()


picar.setup(board, trigger_pin, echo_pin, servo_pin, motor, ena, motor_speed)

model = YOLO(model_path, task="detect")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    right_distance = 0
    left_distance = 0
    # time.sleep(0.1)

    distance = picar.get_distance(board, trigger_pin)
    results = model(frame, imgsz=160, max_det=1, conf=0.5)

    # if distance <= 55:
    if distance <= 25:
        picar.stop_motor(board, motor)
        time.sleep(0.2)
        picar.move_backward(board, motor)
        # time.sleep(0.5)
        time.sleep(0.3)
        picar.stop_motor(board, motor)
        # time.sleep(0.5)
        time.sleep(0.3)

        right_distance = picar.look_right(board, trigger_pin, servo_pin)
        time.sleep(0.5)
        left_distance = picar.look_left(board, trigger_pin, servo_pin)
        time.sleep(0.5)

        if right_distance > left_distance:
            picar.turn_right(board, motor)
            picar.stop_motor(board, motor)

        elif right_distance < left_distance:
            picar.turn_left(board, motor)
            picar.stop_motor(board, motor)

        else:
            picar.move_forward(board, motor)
    else:
        picar.move_forward(board, motor)

    if len(results) > 0:
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

                    print("Area: ", (w - x) * (h - y))

                    if (w - x) * (h - y) > 10000:
                        picar.stop_motor(board, motor)
                        print("Found pingpong ball!")
                        sys.exit()
                else:
                    continue
    # cv2.imshow("result", frame)
    if cv2.waitKey(1) == ord("q"):
        picar.stop_motor(board, motor)
        break

cap.release()
cv2.destroyAllWindows()
