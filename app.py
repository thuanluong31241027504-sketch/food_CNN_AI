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
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    .stApp {
        background-color: #ffffff;
    }
    
    html, body, [class*="css"] {
        font-family: 'Courier New', 'SF Mono', monospace;
    }
    
    @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
    }
    
    .blinking-cursor {
        animation: blink 1s step-end infinite;
        display: inline-block;
        width: 10px;
    }
    
    h1 {
        color: #000000;
        font-weight: normal;
        font-family: 'Courier New', monospace;
        font-size: 1.8rem;
        margin-bottom: 0;
    }
    
    h2, h3, h4 {
        color: #000000;
        font-weight: normal;
        font-family: 'Courier New', monospace;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    p, li, span, div, label {
        color: #000000;
        font-family: 'Courier New', monospace;
    }
    
    .stFileUploader > div {
        background-color: #f5f5f5;
        border: 1px solid #000000;
        border-radius: 0px;
    }
    
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 2px solid #000000 !important;
        border-radius: 0px !important;
        padding: 0.5rem 1rem !important;
        font-family: 'Courier New', monospace !important;
        font-weight: bold !important;
        width: 100% !important;
        box-shadow: none !important;
    }
    
    .stButton > button:hover {
        background-color: #000000 !important;
        color: #ffffff !important;
        border: 2px solid #000000 !important;
        cursor: pointer !important;
    }
    
    .stProgress > div > div > div {
        background-color: #000000;
    }
    
    .streamlit-expanderHeader {
        background-color: #f5f5f5;
        border: 1px solid #000000;
        border-radius: 0px;
        font-family: 'Courier New', monospace;
        color: #000000;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border-left: 1px solid #000000;
        border-right: 1px solid #000000;
        border-bottom: 1px solid #000000;
    }
    
    .stAlert {
        background-color: #f5f5f5;
        border-left: 2px solid #000000;
        border-radius: 0px;
    }
    
    .stCaption {
        color: #666666;
        font-family: 'Courier New', monospace;
    }
    
    code {
        background-color: #f5f5f5;
        color: #000000;
    }
    
    .version {
        color: #666666;
        font-size: 0.8rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title with blinking cursor
st.markdown("""
<h1>
    > Vietnam Food Recognition<span class="blinking-cursor">_</span>
</h1>
""", unsafe_allow_html=True)

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

st.markdown(f"""
> rule: support 30 classes | image size {img_size} | RGB format
""")

# Food data with Vietnamese descriptions
FOOD_DATA = {
    'Banh beo': 'Bánh bèo - Bánh trắng mềm, trên rắc tôm chấy, hành phi, ăn với nước mắm chua ngọt',
    'Banh bot loc': 'Bánh bột lọc - Bánh dai trong suốt, nhân tôm thịt, gói lá chuối',
    'Banh can': 'Bánh căn - Bánh nhỏ đổ khuôn, ăn kèm trứng cút, hành phi, nước mắm',
    'Banh canh': 'Bánh canh - Sợi bánh dày dai, nước dùng đậm đà với giò heo, cua, tôm',
    'Banh chung': 'Bánh chưng - Bánh vuông gói lá dong, nhân đậu xanh, thịt mỡ, ăn Tết',
    'Banh cuon': 'Bánh cuốn - Bánh tráng mỏng cuộn chả lụa, mộc nhĩ, chấm nước mắm chua ngọt',
    'Banh duc': 'Bánh đúc - Bánh mềm xốp, ăn nóng với mộc nhĩ, hành phi, hoặc ăn nguội với nước mắm',
    'Banh gio': 'Bánh giò - Bánh nhân thịt băm, mộc nhĩ, củ đậu, gói lá chuối',
    'Banh khot': 'Bánh khọt - Bánh nhỏ giòn rụm, nhân tôm thịt, ăn kèm rau sống, nước mắm',
    'Banh mi': 'Bánh mì - Ổ bánh mì giòn ruột xốp, kẹp pate, thịt, chả, rau, đồ chua',
    'Banh pia': 'Bánh pía - Bánh Sóc Trăng nhân sầu riêng, đậu xanh, trứng muối, vỏ bánh dẻo',
    'Banh tet': 'Bánh tét - Bánh trụ dài, nhân đậu xanh thịt mỡ, ăn Tết miền Nam',
    'Banh trang nuong': 'Bánh tráng nướng - Bánh tráng nướng giòn, phết trứng, hành, tôm khô, thịt băm',
    'Banh xeo': 'Bánh xèo - Bánh vàng giòn, nhân tôm thịt giá đỗ, cuốn rau sống, chấm nước mắm chua cay',
    'Bun bo Hue': 'Bún bò Huế - Bún sợi to, nước dùng cay thơm sả, bò bắp, giò heo, huyết',
    'Bun dau mam tom': 'Bún đậu mắm tôm - Bún lá, đậu phụ chiên, chả cốm, dồi, ăn với mắm tôm chua cay',
    'Bun mam': 'Bún mắm - Nước dùng từ mắm cá linh, cá sặc, ăn kèm bún, rau, cà tím, bông điên điển',
    'Bun rieu': 'Bún riêu - Nước dùng chua thanh từ cà chua, riêu cua đồng, đậu phụ, ăn kèm rau sống',
    'Bun thit nuong': 'Bún thịt nướng - Bún tươi, thịt nướng thơm, chả giò, rau sống, lạc rang, nước mắm chua ngọt',
    'Ca kho to': 'Cá kho tộ - Cá kho nồi đất, thơm mềm, nước kho sánh mặn ngọt, ăn nóng với cơm trắng',
    'Canh chua': 'Canh chua cá linh - Canh chua thanh mát, cá linh, me, cà chua, bạc hà, giá đỗ',
    'Cao lau': 'Cao lầu - Mì vàng dai, thịt xá xíu, giá đỗ, rau thơm, chút nước sốt đặc trưng Hội An',
    'Chao long': 'Cháo lòng - Cháo nấu cùng nội tạng heo (tim, gan, lòng non), hành lá, tiêu, ăn kèm quẩy',
    'Com tam': 'Cơm tấm - Cơm tấm sườn bì chả, ăn với đồ chua, dưa leo, hành lá, nước mắm ngọt',
    'Goi cuon': 'Gỏi cuốn - Tôm thịt, bún, rau sống cuốn bánh tráng, chấm nước mắm pha hoặc tương đen',
    'Hu tieu': 'Hủ tiếu - Sợi hủ tiếu dai, nước dùng trong, thịt băm, tôm, gan, lòng, ăn kèm rau sống',
    'Mi quang': 'Mì Quảng - Mì vàng dai, ít nước, tôm thịt, đậu phộng, bánh tráng, rau sống',
    'Nem chua': 'Nem chua - Nem chua thanh hóa vị chua cay, bọc lá ổi, dùng với tỏi ớt',
    'Pho': 'Phở - Nước dùng trong ngọt xương, bánh phở trắng mềm, thịt bò tái hoặc chín, rau thơm, hành',
    'Xoi xeo': 'Xôi xéo - Xôi nếp vàng ươm, đậu xanh, hành phi, ăn kèm ruốc, chả, hoặc đường'
}

# Instruction with > on each line
st.markdown("""
> upload image (jpg, jpeg, png)
> click predict
> view result & confidence
""")

# Layout
col_left, col_right = st.columns([0.45, 0.55])

with col_left:
    st.markdown("### > upload")
    uploaded_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_image:
        image = Image.open(uploaded_image)
        
        st.image(image, caption="", width=280)
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        target_size = (input_shape[1], input_shape[2])
        image = image.resize(target_size)
        
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        if st.button("> predict"):
            input_name = session.get_inputs()[0].name
            predictions = session.run(None, {input_name: img_array})[0]
            
            idx = np.argmax(predictions[0])
            confidence = float(predictions[0][idx])
            food_name = list(FOOD_DATA.keys())[idx]
            
            st.markdown(f"### > {food_name}")
            st.caption(f"confidence: {confidence:.2%}")
            
            st.markdown("> description")
            st.write(FOOD_DATA[food_name])
            
            st.markdown("> top 5")
            top5_idx = np.argsort(predictions[0])[-5:][::-1]
            for i, idx in enumerate(top5_idx, 1):
                prob = float(predictions[0][idx])
                name = list(FOOD_DATA.keys())[idx]
                st.progress(prob, text=f"{i}. {name} - {prob:.2%}")

with col_right:
    st.markdown("### > supported classes")
    st.caption("30 mon an viet nam | cuon de xem chi tiet")
    
    with st.container():
        for i, (food, desc) in enumerate(FOOD_DATA.items()):
            with st.expander(f"{i+1:02d}. {food}"):
                st.caption(desc)

# Version footer
st.markdown("""
> version 1.0 2026 by Luong Ngoc Thuan Khanh Anh Doan Hung
""")
