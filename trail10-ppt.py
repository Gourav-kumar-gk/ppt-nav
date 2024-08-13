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

# Variables for gesture tracking
last_click_time = 0
click_threshold = 0.3  # seconds

# Flags for gesture states
is_left_clicking = False

# Variables for smoothing and sensitivity
click_distance_threshold = 0.08  # Adjusted based on observed values
back_distance_threshold = 0.06  # Adjusted based on observed values for thumb and pinky

# Flag to ignore left hand
ignore_left_hand = False

# Function to calculate distance between two points
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

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
    return True

# Function to check if a finger is folded
def is_finger_folded(finger_tip, finger_pip, finger_mcp):
    return False

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

# Function to check if thumb and pinky are meeting
def is_thumb_pinky_meeting(hand_landmarks, threshold=0.06):
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        distance_thumb_pinky = calculate_distance(thumb_tip.x, thumb_tip.y, pinky_tip.x, pinky_tip.y)
        return distance_thumb_pinky < threshold
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

    #on screen flags
    left_hand_detected = left_hand_landmarks is not None
    right_hand_detected = right_hand_landmarks is not None
    left_hand_folded = left_hand_landmarks and is_hand_folded(left_hand_landmarks, click_distance_threshold)
    right_hand_folded = right_hand_landmarks and is_hand_folded(right_hand_landmarks, click_distance_threshold)

    if left_hand_detected and left_hand_folded:
        ignore_left_hand = True
    elif right_hand_folded:
        ignore_left_hand = False

    if ignore_left_hand and right_hand_landmarks:
        
        index_finger_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

        distance_index_thumb = calculate_distance(index_finger_tip.x, index_finger_tip.y, thumb_tip.x, thumb_tip.y)

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

        #pink x tumb
        if is_thumb_pinky_meeting(right_hand_landmarks, back_distance_threshold):
            pyautogui.hotkey('alt', 'left')  # trail hotkey function for previous slide
            cv2.putText(image, 'Go Back', (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    status_text = []
    status_text.append(f"Left Hand Detected: {left_hand_detected}")
    status_text.append(f"Right Hand Detected: {right_hand_detected}")
    status_text.append(f"Left Hand Folded: {left_hand_folded}")
    status_text.append(f"Ignoring Left Hand: {ignore_left_hand}")
    status_text.append(f"Right Hand Folded: {right_hand_folded}")

    for i, line in enumerate(status_text):
        cv2.putText(image, line, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
