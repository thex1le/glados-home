#!/usr/bin/env python3
import time
from picamera2 import Picamera2


def main():
    # Initialize Picamera2
    picam2 = Picamera2()

    # Configure for still image capture
    picam2.configure(picam2.create_still_configuration())

    # Start the camera and allow it to warm up
    picam2.start()
    time.sleep(2)  # Wait for camera sensors to stabilize

    # Capture the image and save it to disk
    picam2.capture_file("test.jpg")
    print("Image saved as test.jpg")


if __name__ == "__main__":
    main()
