from pymycobot import MyCobot280
from pymycobot.genre import Angle
import time

# Replace with your port (e.g., 'COM3' for Windows or '/dev/ttyUSB0' for Linux/Mac)
PORT = "/dev/ttyUSB0"
BAUDRATE = 115200  # Default baud rate for myCobot 280

def main():
    # Initialize connection to the myCobot
    print("Connecting to myCobot 280...")
    robbi = MyCobot280(PORT, BAUDRATE)
    time.sleep(2)  # Wait for connection to stabilize

    # Check connection
    version = robbi._version
    if version:
        print(f"Connected! Firmware version: {version}")
    else:
        print("Failed to connect. Check the USB connection.")
        return

    # Disconnect
    print("Disconnecting...")
    robbi.release_all_servos()
    print("Done!")

if __name__ == "__main__":
    main()