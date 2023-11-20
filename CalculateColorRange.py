# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from pythonosc import udp_client
import time

# %%


# Provided RGB hex colors for Green, Blue, and Red lights
green_colors_hex = ['7c854c', 'e4eeb2', 'fbfbf9', 'eaffc0', 'a4c387', '637d4e', 'd1cf6a', '93942c', '304a1d']
blue_colors_hex = ['b88ddf', 'a481c5', '4d1ab3', 'a37bd4', '4d1ab3', '4d1ab3', '512c9d', '8f6bc5', 'f3c567']
red_colors_hex = ['918428', '533400', 'ffe281', '847242', '4b2d11', 'c5b732', 'fcd685']

# Function to convert hex to RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Function to convert RGB to HSV
def rgb_to_hsv(rgb_color):
    normalized_rgb = np.array([[rgb_color]]) / 255.0
    hsv_color = cv2.cvtColor(np.float32(normalized_rgb), cv2.COLOR_RGB2HSV)[0][0] * [180, 255, 255]
    return tuple(map(int, hsv_color))

# Convert the hex colors to RGB and then to HSV
green_colors_hsv = [rgb_to_hsv(hex_to_rgb(color)) for color in green_colors_hex]
blue_colors_hsv = [rgb_to_hsv(hex_to_rgb(color)) for color in blue_colors_hex]
red_colors_hsv = [rgb_to_hsv(hex_to_rgb(color)) for color in red_colors_hex]

# Create a DataFrame to hold the HSV values
df_green = pd.DataFrame(green_colors_hsv, columns=['H', 'S', 'V'])
df_blue = pd.DataFrame(blue_colors_hsv, columns=['H', 'S', 'V'])
df_red = pd.DataFrame(red_colors_hsv, columns=['H', 'S', 'V'])

# Calculate mean, range, and std for each color channel (H, S, V) for each light color
mean_green = df_green.mean()
range_green = df_green.max() - df_green.min()
std_green = df_green.std()

mean_blue = df_blue.mean()
range_blue = df_blue.max() - df_blue.min()
std_blue = df_blue.std()

mean_red = df_red.mean()
range_red = df_red.max() - df_red.min()
std_red = df_red.std()

# Combine the statistics into a DataFrame for display
df_stats = pd.concat({
    'Green': pd.concat([mean_green, range_green, std_green], axis=1, keys=['Mean', 'Range', 'Std']),
    'Blue': pd.concat([mean_blue, range_blue, std_blue], axis=1, keys=['Mean', 'Range', 'Std']),
    'Red': pd.concat([mean_red, range_red, std_red], axis=1, keys=['Mean', 'Range', 'Std'])
}, axis=1)

df_stats


# %%
# Function to draw 'X' marker
def draw_marker(image, center, color, marker_size):
    cv2.line(image, (center[0] - marker_size, center[1] - marker_size), 
             (center[0] + marker_size, center[1] + marker_size), color, 2)
    cv2.line(image, (center[0] - marker_size, center[1] + marker_size), 
             (center[0] + marker_size, center[1] - marker_size), color, 2)

def find_largest_contour(contours):
    max_area = 0
    largest_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour

# %%
# Max/MSP's IP address and port
maxmsp_ip = "10.155.17.104"  # Change this to the IP of your Max/MSP instance
maxmsp_port = 9000       # Change this to the OSC port Max/MSP is listening on
client = udp_client.SimpleUDPClient(maxmsp_ip, maxmsp_port)

# %% [markdown]
# 

# %%
# Initialize video capture
# cap = cv2.VideoCapture('Experiment2/Bright1.MP4')
cap = cv2.VideoCapture(0) # 0 defualt ; 1 is the extra camera that we gonna use for future.
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output/Bright2_out.avi', fourcc, 20.0, (frame_width, frame_height))

# Define HSV color ranges for Green, Blue, Red
green_lower = (40, 40, 40)
green_upper = (90, 255, 255)

blue_lower = (100, 50, 50)
blue_upper = (140, 255, 255)


red_lower1 = (0, 50, 50)
red_upper1 = (10, 255, 255)
red_lower2 = (160, 50, 50)
red_upper2 = (180, 255, 255)

RXT = 0
RYT = 0
GXT = 0
GYT = 0
BXT = 0

while True:  # Replaced the for loop to make it run continuously
    ret, frame = cap.read(0)
    if not ret:  # Break if no frame is captured
        break
    org_frame = frame.copy()

    # Preprocessing
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    blurred[blurred < 100] = 0

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Masking and contour detection for Green
    mask_green = cv2.inRange(hsv, green_lower, green_upper)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour_green = find_largest_contour(contours_green)
    if largest_contour_green is not None:
        # cv2.drawContours(frame, [largest_contour_green], 0, (0, 255, 0), 2)
        draw_marker(frame, tuple(largest_contour_green[0][0]), (0, 255, 0), 10)


    # Masking and contour detection for Blue
    mask_blue = cv2.inRange(hsv, blue_lower, blue_upper)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour_blue = find_largest_contour(contours_blue)
    if largest_contour_blue is not None:
        # cv2.drawContours(frame, [largest_contour_blue], 0, (255, 0, 0), 2)
        draw_marker(frame, tuple(largest_contour_blue[0][0]), (255, 0, 0), 10)

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour_red = find_largest_contour(contours_red)

    if largest_contour_red is not None:
        draw_marker(frame, tuple(largest_contour_red[0][0]), (0, 0, 255), 10)

    try:
        rx, ry = largest_contour_red[0][0] if largest_contour_red is not None else (RXT, RYT)
        gx, gy = largest_contour_green[0][0] if largest_contour_green is not None else (GXT, GYT)
        bx, by = largest_contour_blue[0][0] if largest_contour_blue is not None else (BXT, BYT)

        RXT, RYT = rx, ry
        GXT, GYT = gx, gy
        BXT, BYT = bx, by

        print(f'Red = [{rx}, {ry}], Green = [{gx}, {gy}], Blue = [{bx}, {by}]')
        client.send_message("/coordinates", (int(rx), int(ry), int(gx), int(gy), int(bx), int(by)))

    except:
        print('3 color not detected')

    # Display the processed video feed
    # cv2.imshow('Processed Video', frame)

    # Check for key press and exit the loop if 'q' is pressed
    # key = cv2.waitKey(1) & 0xFF
    # if key == ord("q"):
    #     break

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

