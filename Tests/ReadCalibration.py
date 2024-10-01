import cv2
import numpy as np
import glob

# Define the chessboard dimensions
CHECKERBOARD = (6, 8)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object points and image points
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load images
images = glob.glob('calibration_image_*.jpg')
print(f"Found {len(images)} images for calibration.")
good_image = 0
bad_image = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    good_image += 1
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Optional: Draw and display the corners
        #cv2.drawChessboardCorners(img, CHECKERBOARD, corners, ret)
        #cv2.imshow('img', img)
        #cv2.waitKey(100)
    else:
        bad_image += 1
        print(f"Chessboard corners not found in image: {fname}")

cv2.destroyAllWindows()

# Check if any valid images were found
if len(objpoints) > 0:
    # Perform camera calibration
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    # Output the calibration results
    print("\nCalibration was successful.")
    print(f"Using {good_image} images")
    print(f"{bad_image} images are rejected")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients:\n{dist_coeffs.ravel()}")
    print(f"Reprojection error: {ret}")

    # Save the calibration results
    print(camera_matrix)
    print(dist_coeffs)
    np.save('camera_matrix.npy', camera_matrix)
    np.save('dist_coeffs.npy', dist_coeffs)
else:
    print("\nCalibration failed. No valid images were found.")
    print("Please ensure that the chessboard corners are visible in the images.")