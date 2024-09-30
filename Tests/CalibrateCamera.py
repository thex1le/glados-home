import time
from picamera2 import PiCamera2

# Initialize the camera
camera = PiCamera2(0)

# Set the camera resolution (you can adjust this as needed)
camera.resolution = (1920, 1080)  # Full HD resolution
camera.framerate = 30

# Allow the camera to warm up
print("Warming up the camera...")
time.sleep(2)

try:
    for i in range(30):
        filename = f'calibration_image_{i+1:02d}.jpg'
        print(f"Capturing {filename}")
        camera.capture(filename)
        time.sleep(1)  # Wait for 1 second before capturing the next image
    print("Image capture complete.")
finally:
    # Release the camera resources
    camera.close()
