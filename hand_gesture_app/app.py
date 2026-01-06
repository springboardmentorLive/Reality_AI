import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# --- STEP 1: Setup the new API components ---

# Configuration options for the Hand Landmarker
model_path = 'hand_landmarker.task' # Make sure you have this file downloaded
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing utilities (these still use the 'solutions' namespace)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Function to draw landmarks on the image for visualization
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]

        # Draw the hand landmarks
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            mp.solutions.hands.HAND_CONNECTIONS, # This connection style still works
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
    return annotated_image

# --- STEP 2: Configure the Hand Landmarker for live stream ---

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO, # Use VIDEO mode for synchronous webcam processing
    num_hands=2)

# --- STEP 3: Process the webcam feed ---

# Use the context manager for automatic cleanup
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Convert the frame from BGR to RGB format which MediaPipe expects
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # Perform hand landmarks detection
        # You need a timestamp for video mode, simple millisecond counter works
        timestamp_ms = cv2.getTickCount() // cv2.getTickFrequency() * 1000
        detection_result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

        # Draw landmarks on the frame
        annotated_image = draw_landmarks_on_image(frame, detection_result)
        
        # Display the resulting frame
        cv2.imshow('Hand Tracking', annotated_image)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()