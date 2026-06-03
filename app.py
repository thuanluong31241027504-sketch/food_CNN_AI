import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(
    page_title="Vietnam Food Recognition",
    page_icon="",
    layout="wide"
)

# Custom CSS - White background, black text, Terminal font
st.markdown("""
<style>
    /* Remove default Streamlit styling */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* White background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Terminal font for all text */
    * {
        font-family: 'Courier New', 'SF Mono', 'Monaco', 'Consolas', monospace !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #000000 !important;
        font-weight: normal !important;
    }
    
    /* Regular text */
    p, li, span, div, label {
        color: #000000 !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #f0f0f0 !important;
        border: 1px solid #000000 !important;
        border-radius: 0px !important;
    }
    
    /* Button */
    .stButton > button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0px !important;
        padding: 0.5rem 1rem !important;
        font-weight: normal !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        background-color: #333333 !important;
        color: #ffffff !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #000000 !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        background-color: #f0f0f0 !important;
        border-left: 2px solid #000000 !important;
        border-radius: 0px !important;
        color: #000000 !important;
    }
    
    /* Divider */
    hr {
        border-color: #000000 !important;
        border-width: 1px !important;
    }
    
    /* Caption */
    .stCaption {
        color: #666666 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
        border-right: 1px solid #000000 !important;
    }
    
    /* Code blocks */
    code {
        background-color: #f0f0f0 !important;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# Title with prompt style
st.markdown("# > Vietnam Food Recognition_")
st.markdown("---")

# Model path
MODEL_PATH = "model.onnx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return ort.InferenceSession(MODEL_PATH)
    return None

# Auto load model
session = load_model()

if session is None:
    st.error("> model.onnx not found")
    st.stop()

# Model info
input_shape = session.get_inputs()[0].shape
st.caption(f"input: {input_shape[1]}x{input_shape[2]}x{input_shape[3]} | output: 30 classes")
st.markdown("---")

# Class names and descriptions
FOOD_DATA = {
    'Banh beo': 'Steamed rice cakes topped with shrimp, served with fish sauce',
    'Banh bot loc': 'Tapioca dumplings filled with shrimp and pork',
    'Banh can': 'Small ceramic bowl pancakes with quail eggs',
    'Banh canh': 'Thick noodle soup with pork and shrimp',
    'Banh chung': 'Square sticky rice cake with mung bean and pork',
    'Banh cuon': 'Steamed rice rolls with minced pork and mushrooms',
    'Banh duc': 'Soft rice cake with pork and fried shallots',
    'Banh gio': 'Pyramid-shaped rice dumpling with pork',
    'Banh khot': 'Mini savory pancakes with shrimp',
    'Banh mi': 'Vietnamese baguette sandwich with various fillings',
    'Banh pia': 'Durian or mung bean cake with salted egg',
    'Banh tet': 'Cylindrical sticky rice cake for Tet holiday',
    'Banh trang nuong': 'Grilled rice paper with egg and toppings',
    'Banh xeo': 'Crispy turmeric pancake with shrimp, pork, bean sprouts',
    'Bun bo Hue': 'Spicy beef noodle soup from Hue',
    'Bun dau mam tom': 'Vermicelli with tofu and shrimp paste',
    'Bun mam': 'Fermented fish noodle soup',
    'Bun rieu': 'Crab noodle soup with tomato broth',
    'Bun thit nuong': 'Vermicelli with grilled pork and herbs',
    'Ca kho to': 'Caramelized fish in clay pot',
    'Canh chua': 'Sweet and sour soup with fish and vegetables',
    'Cao lau': 'Hoi An noodles with pork and greens',
    'Chao long': 'Pork congee with offal',
    'Com tam': 'Broken rice with grilled pork and egg',
    'Goi cuon': 'Fresh spring rolls with shrimp, pork, herbs',
    'Hu tieu': 'Noodle soup with various meats and seafood',
    'Mi quang': 'Quang Nam noodles with turmeric and herbs',
    'Nem chua': 'Fermented pork roll with chili and garlic',
    'Pho': 'Iconic beef or chicken noodle soup with herbs',
    'Xoi xeo': 'Sticky rice with mung bean and fried shallots'
}

# Two columns layout
col_left, col_right = st.columns([0.6, 0.4])

with col_left:
    st.markdown("### upload image")
    uploaded_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        
        # Display image in grayscale for minimal look
        gray_image = image.convert('L')
        st.image(gray_image, caption="", use_container_width=False, width=300)
        
        # Process image
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        target_size = (input_shape[1], input_shape[2])
        image = image.resize(target_size)
        
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        st.markdown("---")
        
        if st.button("> predict"):
            input_name = session.get_inputs()[0].name
            predictions = session.run(None, {input_name: img_array})[0]
            
            idx = np.argmax(predictions[0])
            confidence = float(predictions[0][idx])
            food_name = list(FOOD_DATA.keys())[idx]
            
            st.markdown("---")
            st.markdown(f"### > {food_name}")
            st.caption(f"confidence: {confidence:.2%}")
            
            st.markdown("---")
            st.caption("description")
            st.write(FOOD_DATA[food_name])
            
            # Top 5
            st.markdown("---")
            st.caption("top 5")
            top5_idx = np.argsort(predictions[0])[-5:][::-1]
            for i, idx in enumerate(top5_idx, 1):
                prob = float(predictions[0][idx])
                name = list(FOOD_DATA.keys())[idx]
                st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

with col_right:
    st.markdown("### supported foods")
    st.caption("30 vietnamese dishes")
    
    # Scrollable list of foods
    for i, (food, desc) in enumerate(FOOD_DATA.items()):
        with st.expander(f"> {i+1:02d}. {food}"):
            st.caption(desc)

# Footer
st.markdown("---")
st.caption("cnn model | vietnamese food recognition")
