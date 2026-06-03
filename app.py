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

# Custom CSS
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
    
    /* Terminal font */
    html, body, [class*="css"] {
        font-family: 'Courier New', 'SF Mono', monospace;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #000000;
        font-weight: normal;
        font-family: 'Courier New', monospace;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    /* Regular text */
    p, li, span, div, label {
        color: #000000;
        font-family: 'Courier New', monospace;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #f5f5f5;
        border: 1px solid #000000;
        border-radius: 0px;
    }
    
    .stFileUploader > div > div {
        color: #000000;
    }
    
    /* Button */
    .stButton > button {
        background-color: #000000;
        color: #ffffff;
        border: none;
        border-radius: 0px;
        padding: 0.5rem 1rem;
        font-family: 'Courier New', monospace;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #333333;
        color: #ffffff;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #000000;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f5f5f5;
        border: 1px solid #000000;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border-left: 1px solid #000000;
        border-right: 1px solid #000000;
        border-bottom: 1px solid #000000;
        font-family: 'Courier New', monospace;
    }
    
    /* Info/Success boxes */
    .stAlert {
        background-color: #f5f5f5;
        border-left: 2px solid #000000;
        border-radius: 0px;
    }
    
    /* Divider */
    hr {
        border-color: #000000;
        margin: 1rem 0;
    }
    
    /* Caption */
    .stCaption {
        color: #666666;
        font-family: 'Courier New', monospace;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid #000000;
    }
    
    /* Fix font overlap */
    .element-container {
        margin-bottom: 0rem;
    }
    
    blockquote {
        margin: 0;
        padding: 0.5rem 1rem;
        border-left: 2px solid #000000;
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("# > Vietnam Food Recognition_")
st.markdown("---")

# Model path
MODEL_PATH = "model.onnx"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return ort.InferenceSession(MODEL_PATH)
    return None

session = load_model()

if session is None:
    st.error("> model.onnx not found")
    st.stop()

# Get model info
input_shape = session.get_inputs()[0].shape
img_size = f"{input_shape[1]}x{input_shape[2]}"
num_classes = input_shape[1]

# Rule box
st.markdown(f"""
> rule: support 30 classes | image size {img_size} | RGB format
""")
st.markdown("---")

# Food data with descriptions
FOOD_DATA = {
    'Banh beo': 'Steamed rice cakes with shrimp',
    'Banh bot loc': 'Tapioca dumplings with shrimp and pork',
    'Banh can': 'Mini ceramic bowl pancakes',
    'Banh canh': 'Thick noodle soup',
    'Banh chung': 'Square sticky rice cake',
    'Banh cuon': 'Steamed rice rolls',
    'Banh duc': 'Soft rice cake',
    'Banh gio': 'Pyramid rice dumpling',
    'Banh khot': 'Mini savory pancakes',
    'Banh mi': 'Vietnamese baguette sandwich',
    'Banh pia': 'Durian cake with salted egg',
    'Banh tet': 'Cylindrical sticky rice cake',
    'Banh trang nuong': 'Grilled rice paper',
    'Banh xeo': 'Crispy turmeric pancake',
    'Bun bo Hue': 'Spicy beef noodle soup',
    'Bun dau mam tom': 'Vermicelli with tofu and shrimp paste',
    'Bun mam': 'Fermented fish noodle soup',
    'Bun rieu': 'Crab noodle soup',
    'Bun thit nuong': 'Vermicelli with grilled pork',
    'Ca kho to': 'Caramelized fish in clay pot',
    'Canh chua': 'Sweet and sour soup',
    'Cao lau': 'Hoi An noodles',
    'Chao long': 'Pork congee',
    'Com tam': 'Broken rice with grilled pork',
    'Goi cuon': 'Fresh spring rolls',
    'Hu tieu': 'Noodle soup',
    'Mi quang': 'Turmeric noodles',
    'Nem chua': 'Fermented pork roll',
    'Pho': 'Iconic noodle soup',
    'Xoi xeo': 'Sticky rice with mung bean'
}

# Layout
col_left, col_right = st.columns([0.5, 0.5])

with col_left:
    st.markdown("### > upload")
    uploaded_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        
        # Display image
        st.image(image, caption="", width=280)
        
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
            
            st.markdown("---")
            st.caption("top 5")
            top5_idx = np.argsort(predictions[0])[-5:][::-1]
            for i, idx in enumerate(top5_idx, 1):
                prob = float(predictions[0][idx])
                name = list(FOOD_DATA.keys())[idx]
                st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

with col_right:
    st.markdown("### > supported classes")
    st.caption("30 vietnamese dishes")
    
    # Display list as code block
    food_list = "\n".join([f"{i+1:02d}. {name}" for i, name in enumerate(FOOD_DATA.keys())])
    st.code(food_list, language="")

# Footer
st.markdown("---")
st.caption("cnn model | vietnamese food recognition")
