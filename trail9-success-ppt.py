import cv2
import pyautogui
import numpy as np
import mediapipe as mp
import math
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

# Disable PyAutoGUI fail-safe (use with caution)
pyautogui.FAILSAFE = False

# Initialize webcam
cap = cv2.VideoCapture(1)  # Change index if necessary

# Screen dimensions for mouse control
screen_width, screen_height = pyautogui.size()

# Variables for gesture tracking
last_click_time = 0
click_threshold = 0.3  # seconds

# Flags for gesture states
is_right_clicking = False
is_left_clicking = False

# Variables for smoothing and sensitivity
prev_x, prev_y = None, None
smoothing_factor = 5
click_distance_threshold = 0.08  # Adjusted based on observed values

# Flag to ignore left hand
ignore_left_hand = False

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Function to smooth cursor movement
def smooth_movement(current_x, current_y):
    global prev_x, prev_y
    if prev_x is None or prev_y is None:
        prev_x, prev_y = current_x, current_y
    smoothed_x = prev_x + (current_x - prev_x) / smoothing_factor
    smoothed_y = prev_y + (current_y - prev_y) / smoothing_factor
    prev_x, prev_y = smoothed_x, smoothed_y
    return int(smoothed_x), int(smoothed_y)

# Function to detect double-click gesture
def is_double_click(click_time):
    global last_click_time
    if click_time - last_click_time < click_threshold:
        last_click_time = 0  # reset to avoid multiple clicks
        return True
    last_click_time = click_time
    return False

# Function to check if a finger is extended
def is_finger_extended(finger_tip, finger_pip, finger_mcp):
    return finger_tip.y < finger_pip.y < finger_mcp.y

# Function to check if a finger is folded
def is_finger_folded(finger_tip, finger_pip, finger_mcp):
    return finger_tip.y > finger_pip.y

# Function to check if a hand is folded
def is_hand_folded(hand_landmarks, threshold=0.08):
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        distances = [
            calculate_distance(thumb_tip.x, thumb_tip.y, index_tip.x, index_tip.y),
            calculate_distance(thumb_tip.x, thumb_tip.y, middle_tip.x, middle_tip.y),
            calculate_distance(thumb_tip.x, thumb_tip.y, ring_tip.x, ring_tip.y),
            calculate_distance(thumb_tip.x, thumb_tip.y, pinky_tip.x, pinky_tip.y)
        ]

        return all(dist < threshold for dist in distances)
    return False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand_landmarks = None
    right_hand_landmarks = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            if hand_x < 0.5:
                left_hand_landmarks = hand_landmarks
            else:
                right_hand_landmarks = hand_landmarks

    # Update status flags
    left_hand_detected = left_hand_landmarks is not None
    right_hand_detected = right_hand_landmarks is not None
    left_hand_folded = left_hand_landmarks and is_hand_folded(left_hand_landmarks, click_distance_threshold)
    right_hand_folded = right_hand_landmarks and is_hand_folded(right_hand_landmarks, click_distance_threshold)

    if left_hand_detected and left_hand_folded:
        ignore_left_hand = True
    elif right_hand_folded:
        ignore_left_hand = False

    if ignore_left_hand and right_hand_landmarks:
        # Extract coordinates and convert to screen space
        index_finger_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        tip_x = int(index_finger_tip.x * screen_width)
        tip_y = int(index_finger_tip.y * screen_height)
        thumb_x = int(thumb_tip.x * screen_width)
        thumb_y = int(thumb_tip.y * screen_height)

        # Smooth cursor movement
        smoothed_x, smoothed_y = smooth_movement(tip_x, tip_y)

        # Calculate distances for gestures
        distance_index_thumb = calculate_distance(index_finger_tip.x, index_finger_tip.y, thumb_tip.x, thumb_tip.y)

        # Click gesture: index finger tip close to thumb tip and both extended
        if distance_index_thumb < 0.06 and is_finger_extended(index_finger_tip, right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP], right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]) and is_finger_extended(thumb_tip, right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP], right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]):
            is_left_clicking = True
            click_time = time.time()
            if is_double_click(click_time):
                pyautogui.doubleClick()
                cv2.putText(image, 'Double Click', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                pyautogui.click()
                cv2.putText(image, 'Click', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            is_left_clicking = False

    # Display the status
    status_text = []
    status_text.append(f"Left Hand Detected: {left_hand_detected}")
    status_text.append(f"Right Hand Detected: {right_hand_detected}")
    status_text.append(f"Left Hand Folded: {left_hand_folded}")
    status_text.append(f"Ignoring Left Hand: {ignore_left_hand}")
    status_text.append(f"Right Hand Folded: {right_hand_folded}")
    
    for i, line in enumerate(status_text):
        cv2.putText(image, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the image
    cv2.imshow('Hand Tracking', image)

    # Break loop on 'ESC' key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
