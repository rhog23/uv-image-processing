# smart_pan_tilt_system/tracker.py

"""
Core tracking logic that combines vision input with hardware control.
"""
from typing import Tuple
from hardware_controller import HardwareController

# from vision_processor import VisionProcessor # Not directly needed, main handles data flow
import config  # Using direct import for config values


class FaceTracker:
    """
    Implements the logic to track a face using servo motors.
    """

    def __init__(
        self,
        hw_controller: HardwareController,
        primary_servo_pin: int,
        secondary_servo_pin: int,
    ) -> None:
        """
        Initializes the FaceTracker.

        Args:
            hw_controller (HardwareController): Instance of the hardware controller.
            primary_servo_pin (int): Pin for the primary pan servo.
            secondary_servo_pin (int): Pin for the secondary pan servo.
        """
        self.hw_controller: HardwareController = hw_controller
        self.primary_pin: int = primary_servo_pin
        self.secondary_pin: int = secondary_servo_pin

        # Ensure servos are configured in HardwareController
        if self.primary_pin not in self.hw_controller.servo_configs:
            raise ValueError(
                f"Primary servo (pin {self.primary_pin}) not configured in HardwareController."
            )
        if self.secondary_pin not in self.hw_controller.servo_configs:
            raise ValueError(
                f"Secondary servo (pin {self.secondary_pin}) not configured in HardwareController."
            )

    def adjust_horizontal_pan(self, angle_offset_deg: float) -> None:
        """
        Adjusts the horizontal pan servos (primary and secondary) based on the
        calculated angular offset of the target.

        Args:
            angle_offset_deg (float): The angular offset of the target from the
                                      camera's center view, in degrees.
                                      Positive means target is to the right.
        """
        current_pan_primary_us = self.hw_controller.get_servo_current_position(
            self.primary_pin
        )
        current_pan_secondary_us = self.hw_controller.get_servo_current_position(
            self.secondary_pin
        )

        if angle_offset_deg == 0.0:
            # If target is centered or no target, try to return secondary servo to neutral
            if (
                abs(current_pan_secondary_us - config.PAN_NEUTRAL_SECONDARY)
                > config.MAX_STEP
            ):
                # Move secondary towards its neutral if not already there
                self.hw_controller.move_servo_smooth(
                    self.secondary_pin,
                    float(config.PAN_NEUTRAL_SECONDARY),  # Target neutral
                    config.KP,
                    config.MAX_STEP,
                )
            return

        # Convert angle offset to microsecond adjustment
        # Negative sign because a positive angle_offset (target to the right)
        # might require decreasing pulse width for some servo setups, or increasing for others.
        # The original code used -angle_offset * 11. Let's stick to that convention.
        # If target is to the right (positive angle_offset_deg), we want to move right.
        # If moving right means increasing pulse width, microsec_needed should be positive.
        # So, microsec_needed = angle_offset_deg * config.ANGLE_TO_MICROSEC_FACTOR
        # The original code: microsec_needed = -angle_offset * 11
        # This implies that if offset is positive (object to right), servo value needs to decrease.
        # This depends on servo orientation. We'll use the original logic for now.
        microsec_needed: float = -angle_offset_deg * config.ANGLE_TO_MICROSEC_FACTOR

        # 1. Attempt to move with the primary servo
        primary_target_us: float = current_pan_primary_us + microsec_needed

        new_primary_pos_us = self.hw_controller.move_servo_smooth(
            self.primary_pin, primary_target_us, config.KP, config.MAX_STEP
        )

        # Calculate how much of the desired movement the primary servo actually achieved
        primary_movement_achieved_us: float = (
            new_primary_pos_us - current_pan_primary_us
        )

        # Calculate remaining movement needed that primary couldn't cover (due to limits or step size)
        remaining_microsec_needed: float = (
            microsec_needed - primary_movement_achieved_us
        )

        # 2. If there's remaining movement, use the secondary servo
        if (
            abs(remaining_microsec_needed) > 0.5
        ):  # Only if significant remaining movement
            secondary_target_us: float = (
                current_pan_secondary_us + remaining_microsec_needed
            )

            self.hw_controller.move_servo_smooth(
                self.secondary_pin, secondary_target_us, config.KP, config.MAX_STEP
            )
