import time
import cv2
from pymata4 import pymata4
from ultralytics import YOLO

# Define pin mappings
echoPin1 = 0
trigPin1 = 1
echoPin2 = 2
trigPin2 = 3
pinBuzzer1 = 4
pinBuzzer2 = 5
pinServo1 = 6
pinServo2 = 7
pinLed1 = 8
pinLed2 = 9

# Initialize Arduino board
board = pymata4.Pymata4()

# Setup ultrasonic sensors and servos
ultrasonic_pins = [(trigPin1, echoPin1), (trigPin2, echoPin2)]
servo_pins = [pinServo1, pinServo2]
buzzer_pins = [pinBuzzer1, pinBuzzer2]
led_pins = [pinLed1, pinLed2]

# Initialize YOLO model
model = YOLO("models/toy-train-det-01_openvino_model", task="detect")

# Open the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Initial states
cek1, cek2, kedip = False, False, False


def setup(board, servo_pins, ultrasonic_pins, buzzer_pins, led_pins):
    for trig, echo in ultrasonic_pins:
        board.set_pin_mode_sonar(trig, echo)

    for servo in servo_pins:
        board.set_pin_mode_servo(servo)
        board.servo_write(servo, 0)

    for buzzer in buzzer_pins:
        board.set_pin_mode_tone(buzzer)

    for led in led_pins:
        board.set_pin_mode_digital_output(led)


# Function to open barrier
def open_barrier(board, servo_pins):
    for pos in range(90, -1, -1):
        board.servo_write(servo_pins[0], pos)
        board.servo_write(servo_pins[1], pos)
        time.sleep(0.05)


def activate_buzzer(board, buzzer_pins):

    pass


# Function to reset all components
def reset_components(board, led_pins, servo_pins, buzzer_pins):
    for buzzer in buzzer_pins:
        board.play_tone_off(buzzer)

    for servo in servo_pins:
        board.servo_write(servo, 0)

    for led in led_pins:
        board.digital_write(led, 0)


setup(board, servo_pins, ultrasonic_pins)

# Main loop
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, max_det=1)
        if len(results[0].boxes.cls) != 0:
            if results[0].boxes.cls[0] == 0:
                distances = []

                for ultrasonic in ultrasonic_pins:
                    trig, echo = ultrasonic
                    board.sonar_read(trig, echo)
                    time.sleep(0.01)  # Short delay to allow sonar reading
                    distance, _ = board.sonar_read(trig)
                    distances.append(distance)

                jarak1, jarak2 = distances

                print("Jarak 1:", jarak1, "cm")
                print("Jarak 2:", jarak2, "cm")

        # for result in results:
        #     # Check if a train is detected
        #     print(result.boxes.cls)
        #     train_detected = True  # Assuming 'train' is the label for the toy train

        #     if train_detected:
        #         # Get the distances from ultrasonic sensors
        #         distances = []
        #         for trig, echo in ultrasonic_pins:
        #             board.sonar_read(trig, echo)
        #             time.sleep(0.1)  # Short delay to allow sonar reading
        #             distance = board.get_sonar_data(trig)
        #             distances.append(
        #                 distance[1] / 58.0 if distance else float("inf")
        #             )  # Convert microseconds to cm

        #         jarak1, jarak2 = distances

        #         print("Jarak 1:", jarak1, "cm")
        #         print("Jarak 2:", jarak2, "cm")

        #         # Perform actions based on distance readings
        #         if 2 <= jarak1 <= 10 and not cek2 and not cek1:
        #             cek1 = True
        #             kedip = True
        #             move_servo(pinServo1, 90)

        #         if 2 <= jarak2 <= 10 and not cek1 and not cek2:
        #             cek2 = True
        #             kedip = True
        #             move_servo(pinServo1, 90)

        #         if 2 <= jarak2 <= 10 and cek1:
        #             board.no_tone(pinBuzzer1)
        #             board.no_tone(pinBuzzer2)
        #             time.sleep(1.5)
        #             kedip = False
        #             board.digital_write(pinLed1, 0)
        #             board.digital_write(pinLed2, 0)
        #             open_barrier()
        #             cek1 = False
        #             time.sleep(1.5)

        #         if 2 <= jarak1 <= 10 and cek2:
        #             board.no_tone(pinBuzzer1)
        #             board.no_tone(pinBuzzer2)
        #             time.sleep(1.5)
        #             kedip = False
        #             board.digital_write(pinLed1, 0)
        #             board.digital_write(pinLed2, 0)
        #             open_barrier()
        #             cek2 = False
        #             time.sleep(1.5)
        #     else:
        #         # Reset all components if no train is detected
        #         reset_components()

        # # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
