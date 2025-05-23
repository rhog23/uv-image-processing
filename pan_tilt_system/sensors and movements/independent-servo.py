from pymata4 import pymata4
import time

# Initialize the board
board = pymata4.Pymata4()

# Servo pins
SERVO_9 = 9
SERVO_6 = 6
SERVO_11 = 5
SERVO_3 = 3  # New servo pin

# Current positions (1500μs is typically center position)
current_positions = {
    SERVO_9: 1500,
    SERVO_6: 1500,
    SERVO_11: 1500,
    SERVO_3: 1500,  # New servo initial position
}


def setup():
    print("Initializing servos...")
    # Set servo pins to servo mode
    board.set_pin_mode_servo(SERVO_9)
    board.set_pin_mode_servo(SERVO_6)
    board.set_pin_mode_servo(SERVO_11)
    board.set_pin_mode_servo(SERVO_3)  # Initialize new servo

    # Set initial positions
    for pin, pos in current_positions.items():
        board.servo_write(pin, pos)

    print("4-Servo Independent Smooth Control")
    print("Commands:")
    print("  [pin] [value] - Move specific servo (e.g., '6 1500')")
    print("  all [value]   - Move all servos simultaneously")
    print("  q             - Quit")


def move_servo_smoothly(pin, target):
    print(f"Moving servo {pin} to {target} μs")

    while True:
        current = current_positions[pin]
        if current == target:
            break

        # Move 1μs closer to target
        new_pos = current + (1 if target > current else -1)
        board.servo_write(pin, new_pos)
        current_positions[pin] = new_pos
        time.sleep(0.002)  # 2ms delay for smoothness


def move_all_servos_smoothly(target):
    print(f"Moving all servos to {target} μs (simultaneous smooth movement)")

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
            user_input = input("Enter command: ").strip().lower()

            if user_input == "q":
                break

            parts = user_input.split()
            if len(parts) == 2:
                try:
                    if parts[0] == "all":
                        target = int(parts[1])
                        if 500 <= target <= 2500:
                            move_all_servos_smoothly(target)
                        else:
                            print("Error: Value must be between 500-2500 microseconds")
                    else:
                        pin = int(parts[0])
                        target = int(parts[1])
                        if pin in current_positions and 500 <= target <= 2500:
                            move_servo_smoothly(pin, target)
                        else:
                            print(
                                "Error: Invalid pin or value. Pins are 3, 6, 9, 11 and value must be 500-2500"
                            )
                except ValueError:
                    print("Error: Please enter valid numbers")
            else:
                print("Invalid command. Format: '[pin] [value]' or 'all [value]'")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nResetting servos to center position...")
        for pin in current_positions:
            board.servo_write(pin, 1500)
        board.shutdown()


if __name__ == "__main__":
    main()
