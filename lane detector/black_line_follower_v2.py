import cv2, rpicar
import numpy as np
from pymata4 import pymata4

cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)

left_motor_FW = 6
left_motor_BW = 7
right_motor_FW = 9
right_motor_BW = 8

enA_pin = 5
enB_pin = 10
motor_speed = 90

ena_pin = [enA_pin, enB_pin]

board = pymata4.Pymata4()

motor = [left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW]

rpicar.setup(board, motor, ena_pin, motor_speed)

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
    mask = cv2.inRange(blurred_frame, 0, 70)

    contours, hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            print("CX : " + str(cx) + "  CY : " + str(cy))
            if cx >= 120:
                print("Turn Left")
                rpicar.turn_left(board, motor)

            if cx < 120 and cx > 40:
                print("On Track!")
                rpicar.move_forward(board, motor)

            if cx <= 40:
                print("Turn Right")
                rpicar.turn_right(board, motor)

            cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.drawContours(frame, c, -1, (0, 255, 0), 1)
    else:
        print("I don't see the line")
        rpicar.stop_motor(board, motor)

    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
        rpicar.stop_motor(board, motor)
        break

cap.release()
cv2.destroyAllWindows()
