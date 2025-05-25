import cv2
import numpy as np
import time
import json
from collections import Counter
import socket
from scipy import stats as st
import threading


class ColorDetectionSystem:
    def __init__(self, esp_ip="192.168.83.180", port=80):
        self.esp_ip = esp_ip
        self.port = port
        self.client_socket = None
        self.detected_colors = []
        self.start_time = time.time()
        self.robot_status = "idle"  # idle, moving, delivering
        self.last_status_time = time.time()
        self.can_detect_color = True
        self.detection_interval = 3  # seconds
        self.idle_wait_time = 5  # seconds to wait after robot becomes idle

        # Color detection parameters
        self.crop_size = 100
        self.blur_kernel = (3, 3)
        self.threshold_value = 150
        self.min_v_threshold = 10

        # Initialize socket connection
        self.connect_to_esp32()

        # Initialize camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print("[ERROR] Failed to open camera.")
            exit()

        # Start status listener thread
        self.status_thread = threading.Thread(
            target=self.listen_for_status, daemon=True
        )
        self.status_thread.start()

    def connect_to_esp32(self):
        """Establish TCP connection to ESP32"""
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.settimeout(5)  # 5 second timeout
            self.client_socket.connect((self.esp_ip, self.port))
            print("[INFO] Connected to ESP32")
        except Exception as e:
            print(f"[ERROR] Failed to connect to ESP32: {e}")
            exit()

    def listen_for_status(self):
        """Listen for status updates from ESP32 in separate thread"""
        try:
            # Create separate socket for receiving status
            status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            status_socket.connect((self.esp_ip, 81))  # Different port for status
            print("[INFO] Status listener connected")

            while True:
                try:
                    data = status_socket.recv(1024).decode().strip()
                    if data:
                        status_data = json.loads(data)
                        self.update_robot_status(status_data)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"[WARNING] Status listener error: {e}")
                    time.sleep(1)
        except Exception as e:
            print(f"[ERROR] Status listener failed: {e}")

    def update_robot_status(self, status_data):
        """Update robot status and determine if color detection should proceed"""
        old_status = self.robot_status
        self.robot_status = status_data.get("status", "idle")
        self.last_status_time = time.time()

        if old_status != self.robot_status:
            print(f"[INFO] Robot status changed: {old_status} -> {self.robot_status}")

        # Determine if we can detect color
        if self.robot_status == "idle":
            # Wait additional time after becoming idle to ensure robot is stable
            if time.time() - self.last_status_time >= self.idle_wait_time:
                self.can_detect_color = True
            else:
                self.can_detect_color = False
                print(f"[INFO] Waiting {self.idle_wait_time}s after idle status...")
        else:
            self.can_detect_color = False
            print(f"[INFO] Robot is {self.robot_status}, pausing color detection")

    def classify_color(self, hue_value):
        """Classify color based on hue value with improved ranges"""
        color_ranges = {
            "me": [(0, 10), (160, 180)],  # Red (wraps around)
            "ji": [(11, 25)],  # Orange
            "ku": [(26, 40)],  # Yellow
            "hi": [(41, 80)],  # Green
            "bi": [(81, 130)],  # Blue
            "un": [(131, 159)],  # Purple
        }

        for color, ranges in color_ranges.items():
            for hue_min, hue_max in ranges:
                if hue_min <= hue_value <= hue_max:
                    return color

        return "unknown"

    def detect_color_in_frame(self, frame):
        """Improved color detection with better preprocessing"""
        height, width, _ = frame.shape
        center_y, center_x = height // 2, width // 2

        # Crop center region
        cropped_frame = frame[
            center_y - self.crop_size : center_y + self.crop_size,
            center_x - self.crop_size : center_x + self.crop_size,
        ]

        # Convert to HSV
        hsv = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
        hue_channel = hsv[:, :, 0]
        saturation_channel = hsv[:, :, 1]
        value_channel = hsv[:, :, 2]

        # Create mask for colored objects (filter out low saturation and low value)
        saturation_mask = saturation_channel > 50  # Filter out grayscale
        value_mask = value_channel > self.min_v_threshold  # Filter out too dark
        combined_mask = saturation_mask & value_mask

        # Get hue values from masked region
        target_hues = hue_channel[combined_mask]

        if target_hues.size > 100:  # Minimum pixels required
            # Use mode instead of median for better dominant color detection
            try:
                dominant_hue = st.mode(target_hues, keepdims=True).mode[0]
            except:
                dominant_hue = int(np.median(target_hues))

            color_code = self.classify_color(dominant_hue)

            color_names = {
                "me": "Red 🔴",
                "ji": "Orange 🟠",
                "ku": "Yellow 🟡",
                "hi": "Green 🟢",
                "bi": "Blue 🔵",
                "un": "Purple 🟣",
            }

            print(
                f"[INFO] Dominant Hue: {dominant_hue}, Color: {color_names.get(color_code, 'Unknown')}"
            )
            return color_code, cropped_frame, hsv

        print("[WARNING] Insufficient colored pixels detected")
        return None, cropped_frame, hsv

    def send_color_to_esp32(self, color):
        """Send color result to ESP32 with error handling"""
        try:
            message = json.dumps({"color": color, "timestamp": time.time()})
            self.client_socket.sendall((message + "\n").encode())
            print(f"[SUCCESS] Sent to ESP32: {color}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send data: {e}")
            # Try to reconnect
            try:
                self.connect_to_esp32()
                message = json.dumps({"color": color, "timestamp": time.time()})
                self.client_socket.sendall((message + "\n").encode())
                print(f"[SUCCESS] Reconnected and sent: {color}")
                return True
            except:
                print("[ERROR] Reconnection failed")
                return False

    def run(self):
        """Main detection loop"""
        print("[INFO] Starting color detection system...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera")
                break

            # Only detect color if robot is ready
            if self.can_detect_color:
                color_result, cropped_frame, hsv = self.detect_color_in_frame(frame)

                if color_result and color_result != "unknown":
                    self.detected_colors.append(color_result)

                # Display windows
                cv2.imshow("Original", cropped_frame)
                cv2.imshow("Hue Channel", hsv[:, :, 0])
                cv2.imshow("Saturation Channel", hsv[:, :, 1])
                cv2.imshow("Value Channel", hsv[:, :, 2])

                # Send dominant color every detection_interval seconds
                if time.time() - self.start_time >= self.detection_interval:
                    if self.detected_colors:
                        dominant_color = Counter(self.detected_colors).most_common(1)[
                            0
                        ][0]
                        print(
                            f"\n[RESULT] Dominant color over {self.detection_interval}s: {dominant_color}"
                        )

                        success = self.send_color_to_esp32(dominant_color)
                        if success:
                            # Reset detection after successful send
                            self.detected_colors.clear()
                            self.start_time = time.time()
                            self.can_detect_color = False  # Wait for next idle period
                    else:
                        print(
                            f"\n[WARNING] No colors detected in {self.detection_interval}s period"
                        )
                        self.start_time = time.time()

            else:
                # Show status when not detecting
                status_frame = frame.copy()
                cv2.putText(
                    status_frame,
                    f"Robot Status: {self.robot_status.upper()}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    status_frame,
                    "Color Detection: PAUSED",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Status", status_frame)

            # Exit condition
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break

        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        if self.client_socket:
            self.client_socket.close()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup completed")


if __name__ == "__main__":
    # Initialize and run the color detection system
    detector = ColorDetectionSystem(esp_ip="192.168.134.33", port=80)
    detector.run()
