from pyfirmata import Arduino, util
import time

board = Arduino("COM3")
it = util.Iterator(board)
it.start()
trigpin = board.get_pin("d:8:o")
echopin = board.get_pin("d:9:i")

while True:
    trigpin.write(0)
    board.pass_time(0.5)
    trigpin.write(1)
    board.pass_time(0.00001)
    trigpin.write(0)
    limit_start = time.time()

    while echopin.read() != 1:
        if time.time() - limit_start > 1:
            break
        pass

    start = time.time()
    while echopin.read() != 0:
        pass
    stop = time.time()
    time_elapsed = stop - start
    print((time_elapsed) * 34300 / 2)
    board.pass_time(1)
