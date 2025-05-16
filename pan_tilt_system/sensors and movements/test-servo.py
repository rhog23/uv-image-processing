from pymata4 import pymata4
import time

# Initialize the board
board = pymata4.Pymata4()

# Servo pins
SERVO_9 = 9
SERVO_6 = 6
SERVO_11 = 11

# Current positions (1500μs is typically center position)
current_positions = {SERVO_9: 1500, SERVO_6: 1500, SERVO_11: 1500}


def setup():
    print("Initializing servos...")
    # Set servo pins to servo mode
    board.set_pin_mode_servo(SERVO_9)
    board.set_pin_mode_servo(SERVO_6)
    board.set_pin_mode_servo(SERVO_11)

    # Set initial positions
    for pin, pos in current_positions.items():
        board.servo_write(pin, pos)

    print("3-Servo Smooth Control")
    print("Enter microseconds value (500-2500) for all servos")
    print("Example: 1500")


def move_servos_smoothly(target):
    print(f"Moving to {target} μs (simultaneous smooth movement)")

    while True:
        all_reached = True

        # Update each servo position
        for pin in current_positions:
            current = current_positions[pin]
            if current != target:
                all_reached = False
                # Move 1μs closer to target
                new_pos = current + (1 if target > current else -1)
                board.servo_write(pin, new_pos)
                current_positions[pin] = new_pos

        if all_reached:
            break

        time.sleep(0.002)  # 2ms delay for smoothness


def main():
    setup()

    try:
        while True:
            user_input = input("Enter target position (500-2500) or 'q' to quit: ")

            if user_input.lower() == "q":
                break

            try:
                target = int(user_input)
                if 500 <= target <= 2500:
                    move_servos_smoothly(target)
                else:
                    print("Error: Value must be between 500-2500 microseconds")
            except ValueError:
                print("Error: Please enter a valid number")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nResetting servos to center position...")
        for pin in current_positions:
            board.servo_write(pin, 1500)
        board.shutdown()


if __name__ == "__main__":
    main()
