import pyfirmata
from time import sleep
import logging as log


class RPiCar:
    def __init__(self, port="/dev/ttyACM1"):
        self.board = pyfirmata.Arduino(
            port
        )  #  Accessing the Arduino's board through the port
        sleep(2)
        log.info("[info] connection established")

    def setup(self):
        self.ena = self.board.digital[10]
        self.enb = self.board.digital[5]
        self.ena.mode = pyfirmata.PWM
        self.enb.mode = pyfirmata.PWM

        self.motor_a_in1 = self.board.digital[9]
        self.motor_a_in2 = self.board.digital[8]

        self.motor_b_in1 = self.board.digital[7]
        self.motor_b_in2 = self.board.digital[6]

        log.debug("[info] setup success")

    def forward(self, speed=0):

        speed = abs(speed / 100)

        self.motor_a_in1.write(1)
        self.motor_a_in2.write(1)
        self.motor_b_in1.write(1)
        self.motor_b_in2.write(1)

        self.ena.write(speed)
        self.enb.write(speed)

    def motor_check(self):
        log.debug("[info] motor check started!")

        for spd in range(100, 0, -25):
            self.forward(spd)
            sleep(2)

    def cleanup(self):
        self.forward()


if __name__ == "__main__":
    car = RPiCar()
    car.setup()

    car.motor_check()
    car.cleanup()
