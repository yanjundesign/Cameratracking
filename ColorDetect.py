import cv2
import numpy as np

# Define the RGB colors
colors = {
    'red': (223, 0, 18),
    'green': (0, 255, 160),
    'blue': (60, 213, 231)
}

# Define the BGR colors for circle drawing
bgr_colors = {
    'red': (18, 0, 223),
    'green': (160, 255, 0),
    'blue': (231, 213, 60)
}

# Convert RGB to HSV for each color
for key, value in colors.items():
    colors[key] = cv2.cvtColor(np.uint8([[list(value)]]), cv2.COLOR_RGB2HSV)[0][0]

# Start the video capture
cap = cv2.VideoCapture(2)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color, hsv_value in colors.items():
        # Define the range for each color
        lower = np.array([hsv_value[0]-35, 100, 100])
        upper = np.array([hsv_value[0]+35, 255, 255])

        # Create a mask for the color
        mask = cv2.inRange(hsv, lower, upper)

        # Find the contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there are any contours
        if contours:
            # Sort the contours by area and keep the largest one
            contour = max(contours, key=cv2.contourArea)

            # Get the moments of the contour to calculate the centroid
            M = cv2.moments(contour)

            # Avoid division by zero
            if M["m00"] != 0:
                # Calculate the centroid
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Draw the centroid on the frame
                # cv2.circle(frame, (cX, cY), 5, (0, 0, 255), -1)
                cv2.circle(frame, (cX, cY), 20, bgr_colors[color], -1)


    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
