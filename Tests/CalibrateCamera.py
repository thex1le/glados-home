import time
from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2(0)

# Configure the camera for still image capture
config = picam2.create_still_configuration(main={"size": (640, 480)})
picam2.configure(config)

# Start the camera
picam2.start()

# Allow the camera to warm up
print("Warming up the camera...")
time.sleep(2)

try:
    for i in range(30):
        filename = f'calibration_image_{i+1:02d}.jpg'
        print(f"Capturing {filename}")
        # Capture the image and save to file
        picam2.capture_file(filename)
        time.sleep(1)  # Wait for 1 second before capturing the next image
    print("Image capture complete.")
finally:
    # Stop the camera
    picam2.stop()