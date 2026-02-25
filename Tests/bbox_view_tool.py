import cv2
import numpy as np

# Create a black image of size 640x480 pixels
image_width = 640
image_height = 480
black_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

# Bounding box coordinates
bounding_box = {'x1': 67.57967, 'y1': 339.48883, 'x2': 208.07974, 'y2': 479.66272}

# Convert floating point coordinates to integer pixel positions
x1 = int(round(bounding_box['x1']))
y1 = int(round(bounding_box['y1']))
x2 = int(round(bounding_box['x2']))
y2 = int(round(bounding_box['y2']))

# Ensure the coordinates are within the image boundaries
x1 = max(0, min(x1, image_width - 1))
x2 = max(0, min(x2, image_width - 1))
y1 = max(0, min(y1, image_height - 1))
y2 = max(0, min(y2, image_height - 1))

# Draw the bounding box on the image in red color (BGR format)
color = (0, 0, 255)  # Red color in BGR
thickness = 2        # Thickness of the rectangle border
cv2.rectangle(black_image, (x1, y1), (x2, y2), color, thickness)

# Display the image
cv2.imshow('Bounding Box', black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
