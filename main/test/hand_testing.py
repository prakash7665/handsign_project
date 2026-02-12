import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.3,
                       min_tracking_confidence=0.3)

mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

# Tip landmarks indices for each finger
# Thumb: 4, Index: 8, Middle: 12, Ring: 16, Pinky: 20
finger_tips_ids = [4, 8, 12, 16, 20]

prev_count = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert BGR to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers_up = []

            # Thumb (special case: compare x)
            if lm_list[finger_tips_ids[0]][0] > lm_list[finger_tips_ids[0] - 1][0]:
                fingers_up.append(1)
            else:
                fingers_up.append(0)

            # Other 4 fingers
            for tip_id in finger_tips_ids[1:]:
                if lm_list[tip_id][1] < lm_list[tip_id - 2][1]:
                    fingers_up.append(1)
                else:
                    fingers_up.append(0)

            total_fingers = fingers_up.count(1)

            if total_fingers != prev_count:
                print(f"Detected sign: {total_fingers}")
                prev_count = total_fingers
                time.sleep(2)

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
