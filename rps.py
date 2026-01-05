import cv2
import mediapipe as mp
import pyttsx3
import time
import random

engine = pyttsx3.init()
engine.setProperty('rate', 160)

def speak(text):
    engine.say(text)
    engine.runAndWait()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def recognize_gesture(landmarks):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb (right hand)
    fingers.append(1 if landmarks[4].x < landmarks[3].x else 0)

    for tip in tips[1:]:
        fingers.append(1 if landmarks[tip].y < landmarks[tip - 2].y else 0)

    if fingers == [0,0,0,0,0]:
        return "Rock"
    elif fingers == [0,1,1,0,0]:
        return "Scissors"
    elif fingers == [1,1,1,1,1]:
        return "Paper"
    else:
        return None

choices = ["Rock", "Paper", "Scissors"]
last_time = 0
cooldown = 2

result_text = "Show Rock / Paper / Scissors"
detected_text = ""

with mp_hands.Hands(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    max_num_hands=1) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            gesture = recognize_gesture(hand.landmark)
            detected_text = f"Detected: {gesture}" if gesture else "Detected: None"

            if gesture:
                current_time = time.time()
                if current_time - last_time > cooldown:
                    last_time = current_time
                    computer = random.choice(choices)

                    if gesture == computer:
                        outcome = "Draw!"
                    elif (gesture == "Rock" and computer == "Scissors") or \
                         (gesture == "Paper" and computer == "Rock") or \
                         (gesture == "Scissors" and computer == "Paper"):
                        outcome = "You Win!"
                    else:
                        outcome = "You Lose!"

                    result_text = f"You: {gesture} | AI: {computer} | {outcome}"
                    speak(result_text)

        cv2.putText(frame, result_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame, detected_text, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Rock Paper Scissors - Gesture Game", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
