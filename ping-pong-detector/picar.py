import time


def setup(board, trigger_pin, echo_pin, servo_pin, motor, ena) -> None:
    print("[info] sets up sensors and motors")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    # sets up ultrasonic
    board.set_pin_mode_sonar(trigger_pin, echo_pin)

    # sets ena and enb
    for e in ena:
        board.set_pin_mode_pwm_output(e)
        board.pwm_write(e, 100)

    # sets up servo
    board.set_pin_mode_servo(servo_pin)
    board.servo_write(servo_pin, 115)  # align the servo to 115°

    # sets up wheels
    board.set_pin_mode_digital_output(left_motor_FW)
    board.set_pin_mode_digital_output(left_motor_BW)
    board.set_pin_mode_digital_output(right_motor_FW)
    board.set_pin_mode_digital_output(right_motor_BW)


def get_distance(board, trigger_pin):
    time.sleep(0.07)
    distance, _ = board.sonar_read(trigger_pin)

    return distance


def look_right(board, trigger_pin, servo_pin):
    board.servo_write(servo_pin, 50)
    time.sleep(0.5)
    right_distance = get_distance(board, trigger_pin)
    print(f"[info] look right | distance: {right_distance}")
    time.sleep(0.1)
    board.servo_write(servo_pin, 115)

    return right_distance


def look_left(board, trigger_pin, servo_pin):
    board.servo_write(servo_pin, 170)
    time.sleep(0.5)
    left_distance = get_distance(board, trigger_pin)
    print(f"[info] look left | distance: {left_distance}")
    time.sleep(0.1)
    board.servo_write(servo_pin, 115)

    return left_distance


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


def move_backward(board, motor) -> None:
    print("[info] moving backward")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_FW, 0)
    board.digital_pin_write(left_motor_BW, 1)
    board.digital_pin_write(right_motor_BW, 1)


def turn_left(board, motor) -> None:
    print("[info] turns left")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_BW, 1)

    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_FW, 0)

    time.sleep(0.2)
    stop_motor(board, motor)


def turn_right(board, motor) -> None:
    print("[info] turns right")
    left_motor_FW, left_motor_BW, right_motor_FW, right_motor_BW = motor
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 1)

    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_BW, 0)

    time.sleep(0.2)
    stop_motor(board, motor)
