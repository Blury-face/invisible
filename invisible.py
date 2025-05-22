import cv2
import numpy as np
import time

# Webcam capture
cap = cv2.VideoCapture(0)

# Allow the camera to warm up
time.sleep(3)

# Capture background frame
ret, background = cap.read()
if not ret:
    print("Failed to capture background.")
    cap.release()
    exit()

# Flip the background for mirror effect
background = cv2.flip(background, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])

    # Create mask to detect black color
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # Invert mask to get non-black areas
    mask_inv = cv2.bitwise_not(black_mask)

    # Keep non-black areas from current frame
    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)

    # Take only black areas from background
    res2 = cv2.bitwise_and(background, background, mask=black_mask)

    # Combine both results
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Show the output
    cv2.imshow("Avada Kedavra", final_output)

    # Exit on 'Esc' key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

