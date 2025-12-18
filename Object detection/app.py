import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Set page configuration
st.set_page_config(page_title="Object Detection App (Torchvision)", layout="wide")

st.title("Object Detection with Torchvision (Faster R-CNN)")
st.write("Upload an image to detect objects.")

# COCO Class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

@st.cache_resource
def load_model():
    # Load a pre-trained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    return model, weights.transforms()

try:
    with st.spinner("Loading model... (this may take a moment)"):
        model, preprocess = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner('Detecting objects...'):
        # Preprocess
        img_tensor = preprocess(image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            prediction = model(img_tensor)[0]
        
        # Filter results
        boxes = prediction['boxes'].cpu().numpy()
        labels = prediction['labels'].cpu().numpy()
        scores = prediction['scores'].cpu().numpy()
        
        draw = ImageDraw.Draw(image)
        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

        counts = {}
        
        for i, score in enumerate(scores):
            if score > 0.5: # Confidence threshold
                box = boxes[i]
                label_idx = labels[i]
                label_name = COCO_INSTANCE_CATEGORY_NAMES[label_idx]
                
                counts[label_name] = counts.get(label_name, 0) + 1
                
                # Draw box
                draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
                
                # Draw label background and text
                text = f"{label_name}: {score:.2f}"
                
                # Get text size using getbbox (left, top, right, bottom)
                text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Draw background rectangle for text
                draw.rectangle(
                    [(box[0], box[1] - text_height - 4), (box[0] + text_width + 4, box[1])],
                    fill="red"
                )
                
                # Draw text
                draw.text((box[0] + 2, box[1] - text_height - 4), text, fill="white", font=font)

    with col2:
        st.subheader("Detected Objects")
        st.image(image, caption="Result", use_container_width=True)

    # Display counts
    st.subheader("Object Counts")
    if counts:
        for name, count in counts.items():
            st.write(f"- **{name}**: {count}")
    else:
        st.write("No objects detected.")
