import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="SightBeyond - Object Analysis",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants & Config ---
COCO_LABELS = [
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

# --- Custom Styling ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@300;400;700&display=swap');
        
        .main {
            background-color: #0e1117;
            color: #fafafa;
            font-family: 'Roboto', sans-serif;
        }
        
        h1, h2, h3 {
            font-family: 'Orbitron', sans-serif;
            color: #00e5ff;
            text-shadow: 0 0 10px rgba(0, 229, 255, 0.5);
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #00e5ff, #2979ff);
            color: white;
            border: none;
            border-radius: 5px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 229, 255, 0.4);
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 229, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

# --- Logic Class ---
class VisionEngine:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    @st.cache_resource
    def load_neural_net(_self):
        # Using a slightly different naming convention internally
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        net = fasterrcnn_resnet50_fpn(weights=weights)
        net.eval()
        return net.to(_self.device), weights.transforms()

    def detect(self, image_pil, confidence_threshold):
        model, transform = self.load_neural_net()
        img_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = model(img_tensor)[0]
            
        # Move to CPU for processing
        boxes = outputs['boxes'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        
        results = []
        for i, score in enumerate(scores):
            if score >= confidence_threshold:
                results.append({
                    'box': boxes[i],
                    'label': COCO_LABELS[labels[i]],
                    'score': score
                })
        return results

# --- UI Class ---
class AppInterface:
    def __init__(self):
        self.engine = VisionEngine()
        inject_custom_css()
        
    def render_sidebar(self):
        with st.sidebar:
            st.title("⚙️ System Control")
            st.markdown("---")
            st.write("Configure the vision parameters.")
            
            uploaded_file = st.file_uploader("📂 Input Query Image", type=["jpg", "png", "jpeg"])
            
            threshold = st.slider(
                "🎯 Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Adjust sensitivity of detection."
            )
            
            st.markdown("---")
            st.caption("Powered by PyTorch & Streamlit")
            
            return uploaded_file, threshold

    def draw_annotations(self, image, results):
        draw = ImageDraw.Draw(image)
        # Using a monospaced font if possible, or default
        try:
            # Try to grab a system font or just use default
            font = ImageFont.truetype("arial.ttf", 18) 
        except:
            font = ImageFont.load_default()

        # Cyberpunk colors
        colors = ["#00e5ff", "#00ff9d", "#ff00e6", "#ffde00"]
        
        for i, res in enumerate(results):
            box = res['box']
            label = res['label']
            score = res['score']
            
            color = colors[i % len(colors)]
            
            # Thick neon borders
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=color, width=4)
            
            # Label background
            text = f"{label.upper()} {score:.0%}"
            
            text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            draw.rectangle(
                [(box[0], box[1] - text_h - 8), (box[0] + text_w + 10, box[1])],
                fill=color
            )
            draw.text((box[0] + 5, box[1] - text_h - 6), text, fill="black", font=font)
            
        return image

    def run(self):
        st.title("SIGHT BEYOND // VISUAL ANALYZER")
        st.markdown("Automated Neural Object Identification System")
        
        file, threshold = self.render_sidebar()
        
        if file:
            original_image = Image.open(file).convert("RGB")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Raw Input")
                st.image(original_image, use_container_width=True)
                
            with st.spinner("Processing neural pathways..."):
                results = self.engine.detect(original_image, threshold)
                
            annotated_image = self.draw_annotations(original_image.copy(), results)
            
            with col2:
                st.subheader("Analysis Result")
                st.image(annotated_image, use_container_width=True)
                
            # Metrics Section
            st.markdown("---")
            st.subheader("Detection Metrics")
            
            if results:
                # Count occurrences
                counts = {}
                for res in results:
                    counts[res['label']] = counts.get(res['label'], 0) + 1
                    
                # Display dynamic metrics in columns
                m_cols = st.columns(min(len(counts), 4))
                for idx, (label, count) in enumerate(counts.items()):
                    with m_cols[idx % 4]:
                        st.metric(label=label.title(), value=count, delta="Detected")
            else:
                st.info("No objects detected at current threshold.")

# --- Main Entry ---
if __name__ == "__main__":
    app = AppInterface()
    app.run()
