# smart_pan_tilt_system/main.py

"""
Main application for the Smart Pan Tracking System.
Initializes components, runs the tracking loop, and handles cleanup.
"""
import cv2
import time
import sys
from pymata4 import pymata4

import config  # Make sure config.py is in the same directory or PYTHONPATH
from hardware_controller import HardwareController
from vision_processor import VisionProcessor  # This now uses the unified detector
from tracker import FaceTracker


def main_loop() -> None:
    """
    Main operational loop for face tracking.
    """
    board = None
    hw_controller = None
    vision = None

    try:
        # 1. Initialize Arduino connection
        print("Initializing Arduino board via Pymata4...")
        try:
            # Attempt to auto-detect COM port.
            # If this fails, you might need to specify it, e.g., board = pymata4.Pymata4(com_port='COM3')
            board = pymata4.Pymata4(com_port=None)
            print("Arduino board connected.")
        except RuntimeError as e:
            print(
                f"Error: Could not connect to Arduino. Is it plugged in and Firmata flashed? {e}"
            )
            print(
                "Try specifying COM port in pymata4.Pymata4(com_port='COMX') or '/dev/ttyACMX'"
            )
            print(
                "Ensure StandardFirmataPlus or FirmataExpress is uploaded to your Arduino."
            )
            return  # Exit if board connection fails

        # 2. Initialize Hardware Controller
        print("Initializing Hardware Controller...")
        hw_controller = HardwareController(board)
        hw_controller.setup_servo(
            config.SERVO_PAN_PRIMARY_PIN,
            config.PAN_NEUTRAL_PRIMARY,
            config.PAN_PRIMARY_RANGE,
        )
        hw_controller.setup_servo(
            config.SERVO_PAN_SECONDARY_PIN,
            config.PAN_NEUTRAL_SECONDARY,
            config.PAN_SECONDARY_RANGE,
        )
        time.sleep(1)  # Allow servos to initialize and settle

        # 3. Initialize Vision Processor (now uses unified detection)
        print("Initializing Vision Processor with Unified Detection...")
        vision = VisionProcessor()  # Uses defaults from config.py
        # including cascade paths and NMS settings.
        print(
            f"Dead Zone configured: +/-{vision.dead_zone_px} pixels (Total: {vision.dead_zone_px*2}px)"
        )

        # 4. Initialize Face Tracker
        print("Initializing Face Tracker...")
        tracker = FaceTracker(
            hw_controller, config.SERVO_PAN_PRIMARY_PIN, config.SERVO_PAN_SECONDARY_PIN
        )

        print("\nStarting Smart Pan Tracking System...")
        print("Press 'q' in the OpenCV window to quit.")

        # ===================== TRACKING LOOP =====================
        while True:
            ret, frame = vision.get_frame()
            if not ret or frame is None:
                print("Error: Could not read frame from camera. Exiting.")
                break

            # Detect faces using the unified approach (frontal, profile, NMS)
            all_detected_faces = vision.detect_faces(frame)

            # Get the largest face from the NMS-filtered list to track
            largest_face_bbox = vision.get_largest_face(all_detected_faces)

            pan_angle_offset_deg = 0.0
            if largest_face_bbox:
                pan_angle_offset_deg = vision.calculate_pan_offset_angle(
                    largest_face_bbox
                )
                tracker.adjust_horizontal_pan(pan_angle_offset_deg)
            else:
                # No face detected (or no face large enough to be considered primary target)
                # Ensure secondary servo tries to return to neutral if no target
                tracker.adjust_horizontal_pan(0.0)

            # Prepare servo positions for display
            servo_display_info = {
                config.SERVO_PAN_PRIMARY_PIN: hw_controller.get_servo_current_position(
                    config.SERVO_PAN_PRIMARY_PIN
                ),
                config.SERVO_PAN_SECONDARY_PIN: hw_controller.get_servo_current_position(
                    config.SERVO_PAN_SECONDARY_PIN
                ),
            }

            # Draw information on the frame
            # Pass all_detected_faces for general visualization and largest_face_bbox for the tracked one
            vision.draw_tracking_info(
                frame, largest_face_bbox, all_detected_faces, servo_display_info
            )

            cv2.imshow("Smart Pan Tracking (Unified Detector)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit signal received.")
                break

            # A small delay can help manage CPU usage and allow servos time to react.
            # Adjust as needed; too long a delay makes tracking less responsive.
            time.sleep(0.02)

    except IOError as e:
        print(f"IOError: {e}")
    except RuntimeError as e:  # Catches Pymata4 specific errors too
        print(f"RuntimeError: {e}")
    except KeyboardInterrupt:
        print("\nShutting down by user request (Ctrl+C)...")
    except FileNotFoundError as e:  # For missing cascade files
        print(
            f"FileNotFoundError: {e}. Please ensure cascade files are correctly placed."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nInitiating cleanup...")
        if vision:
            vision.release()
        cv2.destroyAllWindows()

        if hw_controller:  # hw_controller handles board shutdown internally
            hw_controller.shutdown()
        elif (
            board
        ):  # If hw_controller didn't init, but board did, shut down board directly
            print(
                "Shutting down Arduino board directly (hw_controller might not have initialized)."
            )
            # Ensure servos are at neutral before direct board shutdown if possible
            try:
                board.servo_write(
                    config.SERVO_PAN_PRIMARY_PIN, config.PAN_NEUTRAL_PRIMARY
                )
                board.servo_write(
                    config.SERVO_PAN_SECONDARY_PIN, config.PAN_NEUTRAL_SECONDARY
                )
                time.sleep(0.5)
            except Exception as servo_err:
                print(
                    f"Could not set servos to neutral during direct board shutdown: {servo_err}"
                )
            board.shutdown()

        print("System shutdown complete.")


if __name__ == "__main__":
    main_loop()
