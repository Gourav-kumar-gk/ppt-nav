import cv2
import pyautogui
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)

# Initialize variables
prev_x, prev_y = None, None
smoothing_factor = 0.5
normal_sensitivity = 3.0
minute_sensitivity = 5.0
normal_threshold = 5.0
minute_threshold = 1.0
click_threshold = 0.08  # Adjusted based on observed values

# Get screen size
screen_width, screen_height = pyautogui.size()

# Disable PyAutoGUI fail-safe (use with caution)
pyautogui.FAILSAFE = False

def smooth(prev, current, factor):
    return prev * (1 - factor) + current * factor if prev is not None else current

def adjust_sensitivity(dx, dy, sensitivity):
    speed = np.sqrt(dx**2 + dy**2)
    return sensitivity * (speed / 10.0)

def distance_between_points(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def is_hand_folded(hand_landmarks, threshold=0.08):
    if hand_landmarks:
        # Calculate distances between fingertips for the hand
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        # Compute distances between thumb tip and other fingertips
        distances = [
            distance_between_points([thumb_tip.x, thumb_tip.y], [index_tip.x, index_tip.y]),
            distance_between_points([thumb_tip.x, thumb_tip.y], [middle_tip.x, middle_tip.y]),
            distance_between_points([thumb_tip.x, thumb_tip.y], [ring_tip.x, ring_tip.y]),
            distance_between_points([thumb_tip.x, thumb_tip.y], [pinky_tip.x, pinky_tip.y])
        ]

        # Debug print distances
        print("Distances from thumb tip to other fingertips:", distances)

        # Check if all distances are below the threshold
        return all(dist < threshold for dist in distances)
    return False

# Use the correct source for DroidCam
cap = cv2.VideoCapture(1)  # Change index if necessary

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally to mirror it
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    left_hand_landmarks = None
    right_hand_landmarks = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            if hand_x < 0.5:  # Simple heuristic: left hand is on the left side of the screen
                left_hand_landmarks = hand_landmarks
            else:
                right_hand_landmarks = hand_landmarks

    # Initialize indicators
    left_hand_status = "Hand Not Folded"
    click_performed = False

    # Check left hand state (folded)
    if left_hand_landmarks:
        if is_hand_folded(left_hand_landmarks, click_threshold):
            left_hand_status = "Hand Folded"

    # Check right hand gestures (index finger and thumb touching)
    if right_hand_landmarks:
        thumb_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = right_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Compute coordinates
        h, w, _ = frame.shape
        thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
        index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

        thumb_index_distance = distance_between_points([thumb_x, thumb_y], [index_x, index_y])
        print(f"Distance between thumb and index finger: {thumb_index_distance}")

        if thumb_index_distance < click_threshold:
            if left_hand_landmarks and left_hand_status == "Hand Folded":
                pyautogui.click()  # Perform a click
                click_performed = True
                print("Click Performed")

        # Smooth the cursor movement
        if prev_x is not None and prev_y is not None:
            smoothed_x = smooth(prev_x, index_x, smoothing_factor)
            smoothed_y = smooth(prev_y, index_y, smoothing_factor)
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

        prev_x, prev_y = index_x, index_y

        # Draw landmarks and connections
        if right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Draw status on the frame
    cv2.putText(frame, f"Left Hand: {left_hand_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if click_performed:
        cv2.putText(frame, "Click Performed!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Hand Tracking', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
