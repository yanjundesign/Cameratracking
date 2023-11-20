import cv2
import numpy as np
import operator

# Define the RGB colors
colors_rgb = {
    'red': (223, 0, 18),
    'green': (0, 255, 160),
    'blue': (60, 213, 231)
}

# Define BGR colors for circle drawing
colors_bgr = {
    'red': (18, 0, 223),
    'green': (160, 255, 0),
    'blue': (231, 213, 60)
}

def find_brightest_position(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur the image to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Find the brightest spot
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(blurred)
    return maxLoc

def closest_color(pixel):
    # Calculate the minimum Euclidean distance and closest label
    min_distance = float('inf')
    closest = None
    for name, color in colors_rgb.items():
        distance = sum((p-c)**2 for p, c in zip(pixel, color))
        if distance < min_distance:
            min_distance = distance
            closest = name
    return closest

def calculate_average_color(image, position):
    x, y = position
    # Extract a 50x50 block around the brightest spot
    radius = 25
    # block = image[max(y-25, 0):min(y+25, frame.shape[0]), max(x-25, 0):min(x+25, frame.shape[1])]
    block = image[max(y-radius, 0):min(y+radius, frame.shape[0]), max(x-radius, 0):min(x+radius, frame.shape[1])]
    # Calculate the average color of the extracted block
    average_color_per_row = np.average(block, axis=0)
    average_color = np.average(average_color_per_row, axis=0)
    return average_color

path = 'Dark2.mp4'
cap = cv2.VideoCapture(path)
# cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    detected = []

    for _ in range(3):
        position = find_brightest_position(frame)
        avg_color = calculate_average_color(frame, position)
        color_name = closest_color(avg_color)
        detected.append((position, color_name))
        x, y = position
        # Exclude 50x50 area of this detected region from computation
        frame[max(y-25, 0):min(y+25, frame.shape[0]), max(x-25, 0):min(x+25, frame.shape[1])] = 0

    for position, color_name in detected:
        # cv2.circle(frame, position, 50, colors_bgr[color_name], -1)
        # Draw Coordinate with + sign
        cv2.line(frame, (position[0]-50, position[1]), (position[0]+50, position[1]), colors_bgr[color_name], 2)
        cv2.line(frame, (position[0], position[1]-50), (position[0], position[1]+50), colors_bgr[color_name], 2)
        print(f"{color_name} light at position {position}")

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
