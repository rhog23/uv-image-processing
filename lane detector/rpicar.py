import time


def setup(board, motor, ena, motor_speed) -> None:
    print("[info] sets up sensors and motors")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor

    # sets ena and enb
    for e in ena:
        board.set_pin_mode_pwm_output(e)
        board.pwm_write(e, motor_speed)

    # sets up wheels
    board.set_pin_mode_digital_output(left_motor_FW)
    board.set_pin_mode_digital_output(left_motor_BW)
    board.set_pin_mode_digital_output(right_motor_FW)
    board.set_pin_mode_digital_output(right_motor_BW)


def move_forward(board, motor) -> None:
    print("[info] moving forward")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_BW, 0)


def stop_motor(board, motor) -> None:
    print("[info] stop")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_FW, 0)
    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_BW, 0)


def turn_left(board, motor) -> None:
    print("[info] turns left")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_BW, 1)

    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_FW, 0)



def turn_right(board, motor) -> None:
    print("[info] turns right")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 1)

    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_BW, 0)
