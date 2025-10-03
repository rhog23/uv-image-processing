import time
import json
import socket
import threading


class ColorDetectionSystem:
    def __init__(self, esp_ip="192.168.83.180", port=80):
        self.esp_ip = esp_ip
        self.port = port
        self.client_socket = None
        self.robot_status = "idle"  # idle, moving, delivering
        self.last_status_time = time.time()
        self.can_send_color = True  # Renamed from can_detect_color
        self.send_interval = 3  # seconds, interval for sending user-inputted color (though primarily driven by user input now)
        self.idle_wait_time = 5  # seconds to wait after robot becomes idle

        # Initialize socket connection
        self.connect_to_esp32()

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
        """Update robot status and determine if color sending should proceed"""
        old_status = self.robot_status
        self.robot_status = status_data.get("status", "idle")
        self.last_status_time = time.time()

        if old_status != self.robot_status:
            print(f"[INFO] Robot status changed: {old_status} -> {self.robot_status}")

        # Determine if we can send color
        if self.robot_status == "idle":
            # Wait additional time after becoming idle to ensure robot is stable
            if time.time() - self.last_status_time >= self.idle_wait_time:
                self.can_send_color = True
            else:
                self.can_send_color = False
                print(f"[INFO] Waiting {self.idle_wait_time}s after idle status...")
        else:
            self.can_send_color = False
            print(f"[INFO] Robot is {self.robot_status}, pausing color sending")

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
        """Main loop for user input and sending colors"""
        print("[INFO] Starting color input system...")
        print(
            "Available color codes: me (Red), ji (Orange), ku (Yellow), hi (Green), bi (Blue), un (Purple)"
        )

        while True:
            if self.can_send_color:
                user_input = (
                    input("Enter color code (me, ji, ku, hi, bi, un) or 'q' to quit: ")
                    .strip()
                    .lower()
                )

                if user_input == "q":
                    break

                # Validate user input
                valid_colors = ["me", "ji", "ku", "hi", "bi", "un"]
                if user_input in valid_colors:
                    print(f"[INFO] User entered: {user_input}")
                    success = self.send_color_to_esp32(user_input)
                    if success:
                        self.can_send_color = (
                            False  # Pause sending until robot is idle again
                        )
                        print(
                            f"[INFO] Color '{user_input}' sent. Waiting for robot to become idle again."
                        )
                else:
                    print(
                        "[WARNING] Invalid color code. Please use one of: me, ji, ku, hi, bi, un."
                    )
            else:
                print(
                    f"[INFO] Color sending paused. Robot status: {self.robot_status}. Waiting for robot to become idle to send new color."
                )
                time.sleep(1)  # Small delay to prevent busy-waiting

        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.client_socket:
            self.client_socket.close()
        print("[INFO] Cleanup completed")


if __name__ == "__main__":
    # Initialize and run the color input system
    detector = ColorDetectionSystem(esp_ip="172.20.10.2", port=80)
    detector.run()
