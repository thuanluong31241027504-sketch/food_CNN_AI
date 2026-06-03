import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="Vietnamese Food CNN", page_icon="🍜", layout="wide")

st.title("🍜 Vietnamese Food Recognition with CNN")
st.markdown("---")

# 30 món ăn Việt Nam (thay bằng classes thực tế của model bạn)
CLASS_NAMES = [
    'Banh Xeo', 'Bun Cha', 'Pho', 'Com Tam', 'Goi Cuon',
    'Bun Bo Hue', 'Cao Lau', 'Mi Quang', 'Cha Ca', 'Banh Mi',
    'Xoi', 'Chao Long', 'Banh Cuon', 'Bun Rieu', 'Hu Tieu',
    'Lau', 'Bo Luc Lac', 'Ga Tan', 'Vit Quay', 'Heo Quay',
    'Tom Hum', 'Muc Chien', 'Cua Rang', 'So Diep', 'Bao Ngu',
    'Rau Muong', 'Bap Xao', 'Dua Leo', 'Ca Tim', 'Nam'
]

@st.cache_resource
def load_onnx_model():
    """Load model ONNX"""
    model_path = "model.onnx"
    if os.path.exists(model_path):
        return ort.InferenceSession(model_path)
    return None

# Sidebar
with st.sidebar:
    st.header("⚙️ Thông tin")
    st.success("✅ Model ONNX đã sẵn sàng")
    st.write("🎯 Nhận diện món ăn Việt Nam")
    
    # Upload model (nếu cần)
    uploaded_model = st.file_uploader("Upload model.onnx", type=['onnx'])
    if uploaded_model:
        with open("model.onnx", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ Đã upload model mới!")
        st.rerun()

# Load model
onnx_session = load_onnx_model()

if onnx_session is not None:
    st.success("✅ Model đã được load thành công!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload ảnh món ăn")
        uploaded_file = st.file_uploader("Chọn ảnh...", type=['jpg', 'jpeg', 'png', 'webp'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh của bạn", width=300)
    
    with col2:
        if uploaded_file:
            if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    # Tiền xử lý ảnh
                    img = image.resize((128, 128))
                    img_array = np.array(img).astype(np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    
                    # Dự đoán với ONNX
                    input_name = onnx_session.get_inputs()[0].name
                    predictions = onnx_session.run(None, {input_name: img_array})[0]
                    
                    predicted_idx = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    st.success(f"### 🎯 Kết quả: **{CLASS_NAMES[predicted_idx]}**")
                    st.info(f"📊 Độ tin cậy: **{confidence:.2%}**")
                    
                    # Hiển thị top 5
                    st.write("### 🏆 Top 5 dự đoán:")
                    top5_idx = np.argsort(predictions[0])[-5:][::-1]
                    for idx in top5_idx:
                        st.progress(predictions[0][idx], 
                                   text=f"{CLASS_NAMES[idx]}: {predictions[0][idx]:.2%}")
    
    # Thông tin model
    with st.expander("📋 Thông tin model"):
        st.write(f"- Input name: {onnx_session.get_inputs()[0].name}")
        st.write(f"- Input shape: {onnx_session.get_inputs()[0].shape}")
        st.write(f"- Output name: {onnx_session.get_outputs()[0].name}")
        st.write(f"- Số classes: {len(CLASS_NAMES)}")

else:
    st.info("👈 Vui lòng upload file model.onnx ở thanh bên trái")

st.markdown("---")
st.caption("🍜 CNN Model - Nhận diện món ăn Việt Nam | Chạy với ONNX Runtime")

