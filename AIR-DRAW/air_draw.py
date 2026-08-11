import cv2
import mediapipe as mp
import numpy as np

# Webcam
cap = cv2.VideoCapture(0)

# Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Canvas
canvas = None

# Colors (BGR)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
current_color = RED

prev_x, prev_y = None, None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Draw top bar
    cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, 0), (w//4, 60), RED, -1)
    cv2.rectangle(frame, (w//4, 0), (w//2, 60), GREEN, -1)
    cv2.rectangle(frame, (w//2, 0), (3*w//4, 60), BLUE, -1)
    cv2.rectangle(frame, (3*w//4, 0), (w, 60), (0, 0, 0), -1)
    cv2.putText(frame, "CLEAR", (3*w//4 + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hand detection
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        # Index finger tip
        ix = int(hand.landmark[8].x * w)
        iy = int(hand.landmark[8].y * h)

        # Finger up check
        index_up = hand.landmark[8].y < hand.landmark[6].y
        middle_up = hand.landmark[12].y < hand.landmark[10].y

        # Color selection
        if iy < 60:
            prev_x, prev_y = None, None
            if ix < w//4:
                current_color = RED
            elif ix < w//2:
                current_color = GREEN
            elif ix < 3*w//4:
                current_color = BLUE
            else:
                canvas = np.zeros_like(frame)

        # Drawing
        elif index_up and not middle_up:
            if prev_x is None:
                prev_x, prev_y = ix, iy
            cv2.line(canvas, (prev_x, prev_y), (ix, iy), current_color, 5)
            prev_x, prev_y = ix, iy
        else:
            prev_x, prev_y = None, None

        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # Merge canvas and frame
    frame = cv2.add(frame, canvas)

    cv2.imshow("Air Writing", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
