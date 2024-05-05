import time, sys
from pymata4 import pymata4

# Ultrasonic Sensor's Pin
trigger_pin = 8  #  digital input 8
echo_pin = 9  #  digital input 9

# Servo's Pin
servo_pin = 10
distance = 30  # initial distance is set to 30 cm

# Wheel's Pin
left_motor_FW = 7
left_motor_BW = 6
right_motor_FW = 5
right_motor_BW = 4


board = pymata4.Pymata4()


def ultrasonic_callback(data):
    global distance
    distance = data[2]
    return distance


def setup() -> None:
    print("[info] sets up sensors and motors")
    # sets up ultrasonic
    board.set_pin_mode_sonar(trigger_pin, echo_pin, ultrasonic_callback)

    # sets up servo
    board.set_pin_mode_servo(servo_pin)
    board.servo_write(servo_pin, 115)  # align the servo to 115°

    # sets up wheels
    board.set_pin_mode_digital_output(left_motor_FW)
    board.set_pin_mode_digital_output(left_motor_BW)
    board.set_pin_mode_digital_output(right_motor_FW)
    board.set_pin_mode_digital_output(right_motor_BW)


def look_right():
    board.servo_write(servo_pin, 50)
    time.sleep(0.3)
    right_distance = board.sonar_read(trigger_pin)
    print(f"[info] look right | distance: {right_distance}")
    board.servo_write(servo_pin, 115)

    return right_distance


def look_left():
    board.servo_write(servo_pin, 170)
    time.sleep(0.3)
    left_distance = board.sonar_read(trigger_pin)
    print(f"[info] look left | distance: {left_distance}")
    board.servo_write(servo_pin, 115)

    return left_distance


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
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_BW, 0)


def turn_left() -> None:
    print("[info] turns left")
    board.digital_pin_write(left_motor_FW, 1)
    board.digital_pin_write(right_motor_BW, 1)

    board.digital_pin_write(left_motor_BW, 0)
    board.digital_pin_write(right_motor_FW, 0)

    time.sleep(0.3)
    stop_motor()


def turn_right() -> None:
    print("[info] turns right")
    board.digital_pin_write(right_motor_FW, 1)
    board.digital_pin_write(left_motor_BW, 1)

    board.digital_pin_write(left_motor_FW, 0)
    board.digital_pin_write(right_motor_BW, 0)

    time.sleep(0.3)
    stop_motor()


setup()

while True:
    try:
        right_distance = 0
        left_distance = 0
        time.sleep(0.3)
        board.sonar_read(trigger_pin)

        if distance <= 45:
            stop_motor()
            time.sleep(0.1)
            move_backward()
            time.sleep(0.3)
            stop_motor()
            time.sleep(0.3)

            right_distance = look_right()
            time.sleep(0.3)
            left_distance = look_left()
            time.sleep(0.3)

            if right_distance > left_distance:
                turn_right()

            elif right_distance < left_distance:
                turn_left()

            else:
                move_forward()
        else:
            move_forward()

    except KeyboardInterrupt:
        board.shutdown()
        sys.exit(0)
