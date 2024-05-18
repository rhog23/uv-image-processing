import cv2, rpicar, sys, time
import numpy as np
from pymata4 import pymata4

cap = cv2.VideoCapture(0)
cap.set(3, 160)
cap.set(4, 120)

ena = 5
enb = 10

# Wheel's Pin
left_motor_FW = 7
left_motor_BW = 6
right_motor_FW = 8
right_motor_BW = 9

board = pymata4.Pymata4()


def setup() -> None:
    print("[info] sets up motors")

    # sets up ena
    board.set_pin_mode_pwm_output(ena)
    board.pwm_write(ena, 255)

    board.set_pin_mode_pwm_output(enb)
    board.pwm_write(enb, 255)

    # sets up wheels
    board.set_pin_mode_digital_output(left_motor_FW)
    board.set_pin_mode_digital_output(left_motor_BW)
    board.set_pin_mode_digital_output(right_motor_FW)
    board.set_pin_mode_digital_output(right_motor_BW)


def move_forward() -> None:
    print("[info] moving forward")
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_BW, 0)


def stop_motor() -> None:
    print("[info] stop")
    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_FW, 0)
    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_BW, 0)


def move_backward() -> None:
    print("[info] moving backward")
    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_FW, 0)
    board.digital_pin_write(left_motor_BW, 1)
    board.digital_pin_write(right_motor_BW, 1)


def turn_right() -> None:
    print("[info] turns right")
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_BW, 1)

    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_FW, 0)

    time.sleep(0.1)
    # stop_motor()


def turn_left() -> None:
    print("[info] turns left")
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 1)

    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_BW, 0)

    time.sleep(0.1)
    # stop_motor()


setup()

while True:
    try:
        # time.sleep(2)
        move_forward()
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (7, 7), 0)
        mask = cv2.inRange(blurred_frame, 0, 120)

        contours, hierarchy = cv2.findContours(mask, 1, cv2.CHAIN_APPROX_NONE)
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(
                    frame,
                    f"CX : {str(cx)} , CY : {str(cy)}",
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )
                if cx >= 100:
                    print("Turn Right")
                    turn_right()

                if cx < 100 and cx > 50:
                    print("On Track!")
                    move_forward()

                if cx <= 50:
                    print("Turn Left")
                    turn_left()

                cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.drawContours(frame, c, -1, (0, 255, 0), 1)
        else:
            print("I don't see the line")
            stop_motor()

        result = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])
        # cv2.imshow("Mask", mask)
        # cv2.imshow("Frame", frame)
        cv2.imshow("Frame", result)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # 1 is the time in ms
            stop_motor()
            break
    except KeyboardInterrupt:
        stop_motor()
        board.shutdown()
        sys.exit(0)

cap.release()
cv2.destroyAllWindows()
