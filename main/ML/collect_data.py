import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Open CSV file
f = open('hand_sign_data.csv', 'w', newline='')
csv_writer = csv.writer(f)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for lm in hand_landmarks.landmark:
                lm_list.extend([lm.x, lm.y, lm.z])
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check for key press to label the data
            key = cv2.waitKey(10)
            if 48 <= key <= 53:  # ASCII for '0' to '5'
                label = key - 48
                print(f"Saved sample for label {label}")
                csv_writer.writerow(lm_list + [label])

    cv2.imshow("Collect Data - Press 0-5, 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

f.close()
cap.release()
cv2.destroyAllWindows()
