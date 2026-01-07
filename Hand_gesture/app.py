import cv2
import mediapipe as mp
import pyttsx3
import time

# -------------------- TTS SETUP --------------------
engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------------------- MEDIAPIPE SETUP --------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

last_gesture = ""
last_time = time.time()

# -------------------- GESTURE LOGIC --------------------
def recognize_gesture(landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (right hand)
    fingers.append(landmarks[tips[0]].x < landmarks[tips[0] - 1].x)

    # Other fingers
    for i in range(1, 5):
        fingers.append(landmarks[tips[i]].y < landmarks[tips[i] - 2].y)

    # Gesture classification
    if fingers == [0, 0, 0, 0, 0]:
        return "Fist"
    elif fingers == [1, 1, 1, 1, 1]:
        return "Open Palm"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Thumbs Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Peace"
    elif fingers == [1, 0, 0, 0, 1]:
        return "Call Me"
    elif fingers == [1, 1, 0, 0, 0]:
        return "L Sign"
    else:
        return "Unknown"

# -------------------- MAIN LOOP --------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            gesture = recognize_gesture(hand_landmarks.landmark)

            cv2.putText(
                frame,
                gesture,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )

            # Speak only if gesture changes
            if gesture != last_gesture and time.time() - last_time > 1.5:
                speak(gesture)
                last_gesture = gesture
                last_time = time.time()

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
