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
scroll_threshold = 0.1  # seconds
last_scroll_time = 0

# Flags for gesture states
is_right_clicking = False
is_left_clicking = False
is_scrolling = False

# Variables for smoothing and sensitivity
prev_x, prev_y = None, None
smoothing_factor = 0.5
normal_sensitivity = 3.0
minute_sensitivity = 5.0
normal_threshold = 5.0
minute_threshold = 1.0
click_distance_threshold = 0.08  # Adjusted based on observed values

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Function to smooth cursor movement
def smooth(prev, current, factor):
    return prev * (1 - factor) + current * factor if prev is not None else current

# Function to adjust sensitivity based on movement speed
def adjust_sensitivity(dx, dy, sensitivity):
    speed = np.sqrt(dx**2 + dy**2)
    return sensitivity * (speed / 10.0)

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

    left_hand_status = "Hand Not Folded"
    click_performed = False

    if left_hand_landmarks:
        if is_hand_folded(left_hand_landmarks, click_distance_threshold):
            left_hand_status = "Hand Folded"

    if right_hand_landmarks:
        thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        tip_x, tip_y = int(index_tip.x * screen_width), int(index_tip.y * screen_height)
        thumb_x, thumb_y = int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height)

        thumb_index_distance = calculate_distance(tip_x, tip_y, thumb_x, thumb_y)

        if thumb_index_distance < click_distance_threshold:
            if left_hand_landmarks and left_hand_status == "Hand Folded":
                pyautogui.click()
                click_performed = True

        if prev_x is not None and prev_y is not None:
            smoothed_x = smooth(prev_x, tip_x, smoothing_factor)
            smoothed_y = smooth(prev_y, tip_y, smoothing_factor)
            dx = smoothed_x - prev_x
            dy = smoothed_y - prev_y

            movement_magnitude = np.sqrt(dx**2 + dy**2)
            if movement_magnitude > normal_threshold:
                dynamic_sensitivity = adjust_sensitivity(dx, dy, normal_sensitivity)
            elif movement_magnitude > minute_threshold:
                dynamic_sensitivity = adjust_sensitivity(dx, dy, minute_sensitivity)
            else:
                dynamic_sensitivity = 0

            if dynamic_sensitivity > 0:
                pyautogui.moveRel(dx * dynamic_sensitivity, dy * dynamic_sensitivity)

        prev_x, prev_y = tip_x, tip_y

        if right_hand_landmarks:
            mp_drawing.draw_landmarks(image, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if left_hand_landmarks:
            mp_drawing.draw_landmarks(image, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(image, f"Left Hand: {left_hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if click_performed:
        cv2.putText(image, "Click Performed!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
