# smart_pan_tilt_system/hardware_controller.py

"""
Controls the servo motors via Arduino using pymata4.
"""
import time
from typing import Dict, Tuple
from pymata4 import pymata4
import numpy as np


class HardwareController:
    """
    Manages communication with the Arduino board and controls servo motors.
    """

    def __init__(self, board: pymata4.Pymata4) -> None:
        """
        Initializes the HardwareController.

        Args:
            board (pymata4.Pymata4): An instance of the Pymata4 board.
        """
        self.board: pymata4.Pymata4 = board
        self.servo_configs: Dict[int, Dict[str, any]] = {}

    def setup_servo(
        self, pin: int, neutral_pos: int, min_max_range: Tuple[int, int]
    ) -> None:
        """
        Sets up a single servo motor.

        Args:
            pin (int): The digital pin number on the Arduino where the servo is connected.
            neutral_pos (int): The neutral position (pulse width in microseconds) for the servo.
            min_max_range (Tuple[int, int]): The minimum and maximum pulse width (microseconds).
        """
        self.board.set_pin_mode_servo(pin)
        self.board.servo_write(pin, neutral_pos)
        self.servo_configs[pin] = {
            "neutral": neutral_pos,
            "range": min_max_range,
            "current_pos": float(
                neutral_pos
            ),  # Store as float for precise calculations
        }
        print(f"Servo on pin {pin} initialized to neutral: {neutral_pos}us")

    def move_servo_smooth(
        self, pin: int, target_pos_us: float, kp: float, max_step: float
    ) -> float:
        """
        Moves a servo smoothly towards the target position using proportional control.

        Args:
            pin (int): The servo pin number.
            target_pos_us (float): The target position in microseconds.
            kp (float): Proportional gain for speed control.
            max_step (float): Maximum change in pulse width per update.

        Returns:
            float: The new current position of the servo in microseconds.
        """
        if pin not in self.servo_configs:
            print(f"Error: Servo on pin {pin} not configured.")
            return self.servo_configs.get(pin, {}).get("current_pos", 0.0)

        config = self.servo_configs[pin]
        current_pos: float = config["current_pos"]
        min_us, max_us = config["range"]

        error: float = target_pos_us - current_pos
        step: float = kp * error
        step = np.clip(step, -max_step, max_step)  # type: ignore

        # Only move if step is significant enough to avoid jitter
        if abs(step) > 0.5:  # Threshold for movement
            new_pos: float = current_pos + step
            new_pos_clipped: float = np.clip(new_pos, min_us, max_us)  # type: ignore

            self.board.servo_write(pin, int(new_pos_clipped))
            config["current_pos"] = new_pos_clipped
            return new_pos_clipped

        return current_pos

    def get_servo_current_position(self, pin: int) -> float:
        """
        Gets the current logical position of the servo.

        Args:
            pin (int): The servo pin number.

        Returns:
            float: The current position in microseconds.
        """
        return self.servo_configs.get(pin, {}).get("current_pos", 0.0)

    def set_servos_to_neutral(self) -> None:
        """
        Sets all configured servos to their neutral positions.
        """
        print("Setting servos to neutral positions...")
        for pin, config in self.servo_configs.items():
            self.board.servo_write(pin, config["neutral"])
            config["current_pos"] = float(config["neutral"])
            time.sleep(0.1)  # Small delay for servo to reach position

    def shutdown(self) -> None:
        """
        Safely shuts down the connection to the Arduino board
        after setting servos to neutral.
        """
        self.set_servos_to_neutral()
        time.sleep(0.5)  # Allow servos to settle
        print("Shutting down Arduino board connection.")
        self.board.shutdown()
