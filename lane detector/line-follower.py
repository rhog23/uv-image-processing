from sre_compile import dis
import time, sys
from pymata4 import pymata4

ena = 5
enb = 10

# Wheel's Pin
left_motor_FW = 6
left_motor_BW = 7
right_motor_FW = 9
right_motor_BW = 8


board = pymata4.Pymata4()


def setup() -> None:
    print("[info] sets up motors")

    # sets up ena
    board.set_pin_mode_pwm_output(ena)
    board.pwm_write(ena, 50)

    board.set_pin_mode_pwm_output(enb)
    board.pwm_write(enb, 80)

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


def turn_left() -> None:
    print("[info] turns left")
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_BW, 1)

    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_FW, 0)

    time.sleep(0.5)
    stop_motor()


def turn_right() -> None:
    print("[info] turns right")
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 1)

    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_BW, 0)

    time.sleep(0.5)
    stop_motor()


setup()

while True:
    try:
        time.sleep(2)
        move_forward()

    except KeyboardInterrupt:
        stop_motor()
        board.shutdown()
        sys.exit(0)
