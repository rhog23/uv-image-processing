from pymata4 import pymata4
import time

class BH1750:
    def __init__(self, board, address=0x23):
        self.board = board
        self.address = address
        self.latest_value = None
        self.data_ready = False
        self.board.set_pin_mode_i2c()
        
    def _i2c_callback(self, data):
        """Callback function for I2C read"""
        # Data format: [pin_type=6, i2c_address, register, [data_bytes], timestamp]
        if data[1] == self.address and len(data[3]) == 2:
            raw_value = (data[3][0] << 8) | data[3][1]
            self.latest_value = raw_value / 1.2  # Convert to lux
            self.data_ready = True

    def read_light(self, mode=0x20):
        """
        Read light intensity from BH1750
        :param mode: Measurement mode (default: 0x20 - One-time H-Resolution)
        :return: Light intensity in lux or None if read fails
        """
        self.data_ready = False
        self.latest_value = None
        
        # Send measurement command
        self.board.i2c_write(self.address, [mode])
        
        # Wait for measurement (120-180ms for high res mode)
        time.sleep(0.18)
        
        # Read 2 bytes of result (register=None for BH1750)
        self.board.i2c_read(self.address, None, 2, callback=self._i2c_callback)
        
        # Wait for callback to complete (with timeout)
        timeout = time.time() + 0.5  # 500ms timeout
        while not self.data_ready and time.time() < timeout:
            time.sleep(0.01)
        
        return self.latest_value

# Example usage
if __name__ == "__main__":
    board = pymata4.Pymata4()
    sensor = BH1750(board)
    
    try:
        while True:
            light_level = sensor.read_light()
            if light_level is not None:
                print(f"Light Level: {light_level:.2f} lux")
            else:
                print("Failed to read light level")
            time.sleep(1)
            
    except KeyboardInterrupt:
        board.shutdown()