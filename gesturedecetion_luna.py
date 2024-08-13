import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam.
cap = cv2.VideoCapture(1)

# Screen dimensions for mouse control.
screen_width, screen_height = pyautogui.size()

# Variables to keep track of gestures.
last_click_time = 0
click_threshold = 0.3  # seconds
scroll_threshold = 0.1  # seconds
last_scroll_time = 0

# Flags for gesture states.
is_right_clicking = False
is_left_clicking = False
is_scrolling = False

# Variables for smoothing cursor movement.
smoothening_factor = 5
prev_x, prev_y = 0, 0

# Function to calculate distance between two points.
def calculate_distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)

# Function to detect double-click gesture.
def is_double_click(click_time):
    global last_click_time
    if click_time - last_click_time < click_threshold:
        last_click_time = 0  # reset to avoid multiple clicks
        return True
    last_click_time = click_time
    return False

# Function to detect scroll gesture.
def is_scroll(scroll_time):
    global last_scroll_time
    if scroll_time - last_scroll_time > scroll_threshold:
        last_scroll_time = scroll_time
        return True
    return False

# Function to check if a finger is extended.
def is_finger_extended(finger_tip, finger_pip, finger_mcp):
    return finger_tip.y < finger_pip.y < finger_mcp.y

# Function to check if a finger is folded.
def is_finger_folded(finger_tip, finger_pip, finger_mcp):
    return finger_tip.y > finger_pip.y

# Function to smooth cursor movement.
def smooth_movement(current_x, current_y):
    global prev_x, prev_y
    smoothed_x = prev_x + (current_x - prev_x) / smoothening_factor
    smoothed_y = prev_y + (current_y - prev_y) / smoothening_factor
    prev_x, prev_y = smoothed_x, smoothed_y
    return int(smoothed_x), int(smoothed_y)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display.
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands.
    results = hands.process(image_rgb)

    # Draw the hand annotations on the image.
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the coordinates of the landmarks.
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            index_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            middle_finger_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            index_finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

            # Convert the coordinates to screen space.
            tip_x = int(index_finger_tip.x * screen_width)
            tip_y = int(index_finger_tip.y * screen_height)
            thumb_x = int(thumb_tip.x * screen_width)
            thumb_y = int(thumb_tip.y * screen_height)
            pinky_x = int(pinky_tip.x * screen_width)
            pinky_y = int(pinky_tip.y * screen_height)

            # Smooth cursor movement.
            smoothed_x, smoothed_y = smooth_movement(tip_x, tip_y)

            # Check if fingers are extended or folded.
            index_extended = is_finger_extended(index_finger_tip, index_finger_pip, index_finger_mcp)
            middle_extended = is_finger_extended(middle_finger_tip, middle_finger_pip, index_finger_mcp)
            thumb_extended = is_finger_extended(thumb_tip, thumb_ip, thumb_mcp)
            pinky_extended = is_finger_extended(pinky_tip, middle_finger_pip, index_finger_mcp)
            index_folded = is_finger_folded(index_finger_tip, index_finger_pip, index_finger_mcp)
            middle_folded = is_finger_folded(middle_finger_tip, middle_finger_pip, index_finger_mcp)

            # Move the mouse cursor smoothly.
            pyautogui.moveTo(smoothed_x, smoothed_y)

            # Calculate distances for gestures.
            distance_index_middle = calculate_distance(index_finger_tip.x, index_finger_tip.y, middle_finger_tip.x, middle_finger_tip.y)
            distance_index_thumb = calculate_distance(index_finger_tip.x, index_finger_tip.y, thumb_tip.x, thumb_tip.y)
            distance_thumb_pinky = calculate_distance(thumb_tip.x, thumb_tip.y, pinky_tip.x, pinky_tip.y)
            distance_thumb_screen_top = thumb_tip.y

            # Click gesture: index and middle fingers close together and both extended.
            if distance_index_middle < 0.04 and index_extended and middle_extended and not is_right_clicking and not is_scrolling:
                is_left_clicking = True
                click_time = time.time()
                if is_double_click(click_time):
                    pyautogui.doubleClick()
                    cv2.putText(image, 'Double Click', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    pyautogui.click()
                    cv2.putText(image, 'Click', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                is_left_clicking = False

            # Right-click gesture: index finger tip close to thumb tip and both extended.
            elif distance_index_thumb < 0.06 and index_extended and thumb_extended and not is_left_clicking and not is_scrolling:
                is_right_clicking = True
                pyautogui.rightClick()
                cv2.putText(image, 'Right Click', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                is_right_clicking = False
            
            # Thumb and pinky click gesture: thumb and pinky close together and both extended.
            elif distance_thumb_pinky < 0.05 and thumb_extended and pinky_extended and not is_left_clicking and not is_right_clicking:
                is_left_clicking = True
                pyautogui.click()
                cv2.putText(image, 'Thumb-Pinky Click', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                is_left_clicking = False
            
            # Scroll gesture: thumb moving up or down and extended.
            elif thumb_extended and is_scroll(time.time()) and not is_left_clicking and not is_right_clicking:
                is_scrolling = True
                if distance_thumb_screen_top < 0.3:  # moving up
                    pyautogui.scroll(10)
                    cv2.putText(image, 'Scroll Up', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif distance_thumb_screen_top > 0.7:  # moving down
                    pyautogui.scroll(-10)
                    cv2.putText(image, 'Scroll Down', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                is_scrolling = False

    # Display the image.
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
