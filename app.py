import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(page_title="NHẬN DIỆN MÓN ĂN VIỆT NAM", page_icon="🍜", layout="centered")

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a0a2e 50%, #0a0a1a 100%);
        min-height: 100vh;
    }
    .title {
        font-family: monospace;
        font-size: 2rem;
        text-align: center;
        color: #c084fc;
        margin-top: 40px;
    }
    .result {
        font-family: monospace;
        font-size: 1.2rem;
        text-align: center;
        color: #c084fc;
        padding: 20px;
        border: 1px solid #c084fc;
        border-radius: 8px;
        margin-top: 20px;
    }
    .stButton > button {
        background: transparent;
        color: #c084fc !important;
        border: 2px solid #c084fc;
        border-radius: 8px !important;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🍜 NHẬN DIỆN MÓN ĂN VIỆT NAM</div>', unsafe_allow_html=True)

MODEL_PATH = "food_model.tflite"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    return None

CLASS_NAMES = [
    "banh_bao", "banh_beo", "banh_canh", "banh_chung", "banh_cuon",
    "banh_khot", "banh_mi", "banh_trang", "banh_xeo", "bun_bo_hue",
    "bun_dau_mam_tom", "bun_moc", "bun_rieu", "bun_thit_nuong", "cao_lau",
    "chao_long", "che", "com_tam", "com_tay_cam", "goi_cuon",
    "hu_tieu", "mi_quang", "pho", "xoi"
]

interpreter = load_model()

if interpreter is None:
    st.error("❌ Không tìm thấy model! Vui lòng kiểm tra file food_model.tflite")
else:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = input_details[0]['shape'][1]
    
    st.success(f"✅ Model đã sẵn sàng! Kích thước: {input_size}x{input_size}")
    
    uploaded_file = st.file_uploader("Chọn ảnh món ăn", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Ảnh của bạn", use_container_width=True)
        
        if st.button("🔮 NHẬN DIỆN"):
            with st.spinner("Đang phân tích..."):
                # SỬA LỖI: Chuyển ảnh sang RGB (3 kênh)
                img = img.convert('RGB')
                img = img.resize((input_size, input_size))
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                interpreter.set_tensor(input_details[0]['index'], img_array)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                predicted_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                food_name = CLASS_NAMES[predicted_idx].replace("_", " ").title()
                
                st.markdown(f"""
                <div class="result">
                    🍽️ KẾT QUẢ: {food_name}<br>
                    📊 ĐỘ TIN CẬY: {confidence*100:.1f}%
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8501))
    st.run(server_port=port, server_address="0.0.0.0")
