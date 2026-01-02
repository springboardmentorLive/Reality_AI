import cv2
import mediapipe as mp  
import pyttsx3
import time

import cv2
import mediapipe as mp
import pyttsx3
import time


#Initialize TTS
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    engine.say(text)
    engine.runAndWait()

#mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def recognize_hand(hand_landmarks):
  tips = [4, 8 , 12, 16, 20]
  finger_states = []

  # Thumb (right hand logic) - Simplified for example, might need more robust logic
  # A common simple check for thumb is comparing tip to MCP joint, or tip to base of thumb
  # Let's assume for a right hand, if the tip is further left than the base, it's open.
  # This is a simplification and might not work for all orientations.
  if hand_landmarks.landmark[tips[0]].x < hand_landmarks.landmark[tips[0]-1].x: # Comparing thumb tip (4) to its base (3) for horizontal movement
      finger_states.append(1)
  else:
      finger_states.append(0)

  # Other fingers (index, middle, ring, pinky)
  for i in range(1, len(tips)): # Iterate from index finger to pinky
    if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i]-2].y: # Comparing tip to PIP joint for vertical movement
      finger_states.append(1)
    else:
      finger_states.append(0)

  # Now check gestures based on finger_states
  if finger_states == [1,1,1,1,1]:
    return "Hi"
  elif finger_states == [1,0,0,0,0]:
    return "Thumbs Up"
  elif finger_states == [0,0,0,0,0]:
    return "Fist"
  elif finger_states == [0,1,0,0,0]:
    return "Ok"
  elif finger_states == [0,1,1,0,0]:
    return "Peace"
  else:
    return "Not Recognized"

gesture_speech = {
    "Hi":"Hello there!",
    "Thumbs Up":"good job",
    "Fist":"Ready!",
    "Ok":"Its fine",
    "Peace":"Peacefull!"
}

last_gesture = ""
last_gesture_time = 0
cooldown = 2 # seconds

with mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands = 1
) as hands:
  while cap.isOpened():
     ret , frame = cap.read()
     if not ret:
        break
     
     frame = cv2.flip(frame, 1)
     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
     results = hands.process(rgb)

     gesture_text ="Show a gesture....."
     
     if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
          mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
          gesture = recognize_hand(hand_landmarks)

          if gesture and gesture != "Not Recognized":
            current_time = time.time()

            if gesture != last_gesture or (current_time - last_gesture_time > cooldown):
              last_gesture = gesture
              last_gesture_time = current_time
              speak(gesture_speech.get(gesture, gesture))
          gesture_text = gesture

     cv2.putText(frame, gesture_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
     cv2.imshow("Hand Gesture Recognition", frame)

     if cv2.waitKey(1) & 0xFF == 27:
                break

cap.release()
cv2.destroyAllWindows()

