import RPi.GPIO as GPIO
import time

in1 = 9  # Motor1  L298 Pin in1
in2 = 8  # Motor1  L298 Pin in1
in3 = 7  # Motor2  L298 Pin in1
in4 = 6  # Motor2  L298 Pin in1

buttonPin1 = 2  # Pin tempat tombol switch pertama terhubung
buttonPin2 = 3  # Pin tempat tombol switch kedua terhubung
buttonPin3 = 4  # Pin tempat tombol switch ketiga terhubung


GPIO.setmode(GPIO.BCM)
GPIO.setup(enA, GPIO.OUT)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setup(enB, GPIO.OUT)
GPIO.setup(buttonPin1, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(buttonPin2, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(buttonPin3, GPIO.IN, pull_up_down=GPIO.PUD_UP)


def forward():
    GPIO.output(in1, GPIO.HIGH)
    GPIO.output(in2, GPIO.LOW)
    GPIO.output(in3, GPIO.LOW)
    GPIO.output(in4, GPIO.HIGH)


pwmA = GPIO.PWM(enA, 1000)
pwmB = GPIO.PWM(enB, 1000)
pwmA.start(70)
pwmB.start(70)

try:
    while True:

        if GPIO.input(buttonPin1) == GPIO.LOW:
            print("Angka 1")
            time.sleep(0.2)
        elif GPIO.input(buttonPin2) == GPIO.LOW:
            print("Angka 2")
            time.sleep(0.2)
        elif GPIO.input(buttonPin3) == GPIO.LOW:
            print("Angka 3")
            time.sleep(0.2)

except KeyboardInterrupt:
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
