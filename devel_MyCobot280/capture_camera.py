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

    robbi.clear_error_information()
    print('is the robot powered on? ', robbi.is_power_on())
    print('Checking for errors: ', robbi.get_error_information())


    print("Moving to home position...")
    robbi.send_angles([-4.39, 18.89, -11.68, -17.13, 30.0, 45], 50)  # Angles in degrees, speed = 50
    time.sleep(3)  # Wait for movement to complete

    robbi.set_gripper_state(0, speed=50)  # open gripper
    time.sleep(3)

    angles = robbi.get_angles()
    print(f"Current joint angles: {angles}")

    print("Done!")

if __name__ == "__main__":
    main()