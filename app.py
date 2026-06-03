import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(
    page_title="vietnamese food cnn",
    page_icon="",
    layout="centered"
)

# Custom CSS - Minimal, black & white, Terminal style
st.markdown("""
<style>
    /* Remove all default Streamlit styling */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Black & white theme */
    .stApp {
        background-color: #000000;
    }
    
    /* Terminal font for all text */
    * {
        font-family: 'Courier New', 'SF Mono', 'Monaco', 'Consolas', monospace !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-weight: normal !important;
        letter-spacing: -0.5px;
    }
    
    /* Regular text */
    p, li, span, div {
        color: #ffffff !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 0px !important;
    }
    
    .stFileUploader > div > div {
        color: #ffffff !important;
    }
    
    /* Button */
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
        border-radius: 0px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: bold !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #ffffff !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo, .stError {
        background-color: #111111 !important;
        border-left: 2px solid #ffffff !important;
        border-radius: 0px !important;
    }
    
    /* Divider */
    hr {
        border-color: #333333 !important;
    }
    
    /* Caption */
    .stCaption {
        color: #666666 !important;
    }
    
    /* Disable emoji in all elements */
    [data-testid="stImage"] {
        filter: grayscale(100%);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# vietnamese food cnn")
st.markdown("---")

# Model path - auto load from current directory
MODEL_PATH = "model.onnx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return ort.InferenceSession(MODEL_PATH)
    return None

# Auto load model
session = load_model()

# Check if model exists
if session is None:
    st.error("model.onnx not found")
    st.stop()

# Class names from dataset
CLASS_NAMES = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh canh', 'Banh chung',
    'Banh cuon', 'Banh duc', 'Banh gio', 'Banh khot', 'Banh mi',
    'Banh pia', 'Banh tet', 'Banh trang nuong', 'Banh xeo', 'Bun bo Hue',
    'Bun dau mam tom', 'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Ca kho to',
    'Canh chua', 'Cao lau', 'Chao long', 'Com tam', 'Goi cuon',
    'Hu tieu', 'Mi quang', 'Nem chua', 'Pho', 'Xoi xeo'
]

# Model info display
input_shape = session.get_inputs()[0].shape
st.caption(f"input: {input_shape[1]}x{input_shape[2]}x{input_shape[3]} | output: 30 classes")

st.markdown("---")

# File uploader
uploaded_image = st.file_uploader("select image", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    # Load and display image
    image = Image.open(uploaded_image)
    
    # Convert to grayscale for display
    gray_image = image.convert('L')
    st.image(gray_image, caption="", use_container_width=False, width=300)
    
    # Process image
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    target_size = (input_shape[1], input_shape[2])
    image = image.resize(target_size)
    
    img_array = np.array(image).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict button
    if st.button("predict"):
        input_name = session.get_inputs()[0].name
        predictions = session.run(None, {input_name: img_array})[0]
        
        idx = np.argmax(predictions[0])
        confidence = float(predictions[0][idx])
        
        st.markdown("---")
        st.markdown(f"### {CLASS_NAMES[idx]}")
        st.caption(f"confidence: {confidence:.2%}")
        
        # Top 5
        st.markdown("---")
        st.caption("top 5")
        top5_idx = np.argsort(predictions[0])[-5:][::-1]
        for i, idx in enumerate(top5_idx, 1):
            prob = float(predictions[0][idx])
            st.progress(prob, text=f"{CLASS_NAMES[idx]} - {prob:.2%}")

# Footer
st.markdown("---")
st.caption("cnn model | vietnamese food recognition")

