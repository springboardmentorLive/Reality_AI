# Hand Gesture Recognition with Speech Feedback

This project works on a real-time Hand Gesture Recognition application using OpenCV and MediaPipe. It tracks hand landmarks, classifies gestures, and provides audio feedback using text-to-speech.

## Features

- **Real-time Hand Tracking**: Uses MediaPipe Hands to detect and track hand landmarks.
- **Gesture Recognition**: Identifies various hand gestures based on finger positions.
  - Fist
  - Open Palm
  - Thumbs Up
  - Peace Sign
  - Call Me
  - L Sign
- **Text-to-Speech**: Announces the recognized gesture using `pyttsx3`.
- **Visual Feedback**: Displays the recognized gesture name on the video feed.

## Prerequisites

Ensure you have Python installed. You will need the following libraries:

- `opencv-python`
- `mediapipe`
- `pyttsx3`

## Installation

1.  **Clone the repository** (or download the source code):
    ```bash
    git clone <repository_url>
    cd Hand_gesture
    ```

2.  **Install the dependencies**:
    ```bash
    pip install opencv-python mediapipe pyttsx3
    ```

## Usage

1.  **Run the application**:
    ```bash
    python app.py
    ```

2.  **Interact**:
    - Show your hand to the camera.
    - Make gestures like a fist, open palm, thumbs up, etc.
    - The application will display the gesture name and speak it out loud.

3.  **Exit**:
    - Press `ESC` key to close the application.

## How it Works

- **Video Capture**: OpenCV captures the video feed from the webcam.
- **Hand Processing**: MediaPipe processes each frame to find hand landmarks.
- **Logic**: A simple heuristic determines the gesture based on which fingers are extended (using landmark coordinates).
- **Feedback**: If the gesture changes and stays stable, the TTS engine speaks the gesture name.

## Code Structure

- `app.py`: Main script containing the setup, gesture recognition logic, and the main loop.

## Troubleshooting

- **Camera not opening**: Ensure no other application is using the webcam.
- **Dependencies**: If you face errors, try upgrading pip and reinstalling the requirements.
