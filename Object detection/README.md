# Streamlit Object Detection App

This is a web application that performs object detection on uploaded images using Deep Learning. It identifies objects in an image, draws bounding boxes around them, and counts the occurrences of each object type.

## How It Works

1.  **User Interface**: Built with **Streamlit**, allowing users to easily upload images via a web browser.
2.  **Image Processing**: The uploaded image is converted to a format suitable for the model using **PIL (Python Imaging Library)** and **Torchvision transforms**.
3.  **Object Detection Model**: The app uses a pre-trained **Faster R-CNN (Region-based Convolutional Neural Network)** model with a ResNet-50-FPN backbone.
    -   **Model Source**: `torchvision.models.detection`
    -   **Weights**: Pre-trained on the COCO dataset (`FasterRCNN_ResNet50_FPN_Weights.DEFAULT`), capable of detecting 91 different classes of objects (e.g., person, car, dog, bicycle).
4.  **Inference**: The model analyzes the image and outputs bounding boxes, labels, and confidence scores for detected objects.
5.  **Visualization**:
    -   The app filters detections based on a confidence threshold (default > 0.5).
    -   It draws bounding boxes and labels directly onto the image using **PIL**.
    -   It displays the processed image and a summary count of detected objects.

## Technologies Used

-   **Python**: The core programming language.
-   **Streamlit**: For building the interactive web interface.
-   **PyTorch & Torchvision**: For loading and running the deep learning model (Faster R-CNN).
-   **Pillow (PIL)**: For image manipulation and drawing.
-   **NumPy**: For numerical operations.

## How to Run

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Application**:
    ```bash
    streamlit run app.py
    ```

3.  **Use the App**:
    -   Open the URL provided in the terminal (usually `http://localhost:8501`).
    -   Upload an image (JPG, PNG).
    -   View the results!
