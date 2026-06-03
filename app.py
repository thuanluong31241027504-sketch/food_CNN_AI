import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="Vietnamese Food CNN", page_icon="🍜", layout="wide")

st.title("🍜 Vietnamese Food Recognition with CNN")
st.markdown("---")

# 30 món ăn Việt Nam
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
        session = ort.InferenceSession(model_path)
        return session
    return None

# Sidebar
with st.sidebar:
    st.header("⚙️ Thông tin model")
    st.write("📥 Input: 224x224x3 (RGB)")
    st.write("📤 Output: 30 classes")
    
    uploaded_model = st.file_uploader("Upload model.onnx", type=['onnx'])
    if uploaded_model:
        with open("model.onnx", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ Đã upload model mới!")
        st.rerun()

# Load model
session = load_onnx_model()

if session is not None:
    st.success("✅ Model đã được load thành công!")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload ảnh món ăn")
        uploaded_file = st.file_uploader("Chọn ảnh...", type=['jpg', 'jpeg', 'png', 'webp'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh gốc", width=250)
            
            # Hiển thị thông tin ảnh gốc
            st.write(f"Kích thước gốc: {image.size}")
            st.write(f"Mode: {image.mode}")
    
    with col2:
        if uploaded_file:
            if st.button("🔍 Dự đoán", type="primary", use_container_width=True):
                with st.spinner("Đang xử lý..."):
                    try:
                        # QUAN TRỌNG: Chuyển sang RGB và resize đúng kích thước
                        if image.mode == 'RGBA':
                            # Chuyển từ RGBA sang RGB (loại bỏ kênh alpha)
                            image = image.convert('RGB')
                            st.write("✅ Đã chuyển từ RGBA → RGB")
                        
                        # Resize về 224x224 (đúng với yêu cầu model)
                        img = image.resize((224, 224))
                        
                        # Chuyển sang numpy array
                        img_array = np.array(img).astype(np.float32)
                        
                        # Normalize về [0,1]
                        img_array = img_array / 255.0
                        
                        # Thêm batch dimension
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        # Kiểm tra shape
                        st.write(f"✅ Shape đầu vào: {img_array.shape}")
                        st.write(f"✅ Kỳ vọng: (1, 224, 224, 3)")
                        
                        # Dự đoán
                        input_name = session.get_inputs()[0].name
                        predictions = session.run(None, {input_name: img_array})[0]
                        
                        predicted_idx = np.argmax(predictions[0])
                        confidence = float(np.max(predictions[0]))
                        
                        st.success(f"### 🎯 Kết quả: **{CLASS_NAMES[predicted_idx]}**")
                        st.info(f"📊 Độ tin cậy: **{confidence:.2%}**")
                        
                        # Hiển thị top 5
                        st.write("### 🏆 Top 5 dự đoán:")
                        top5_idx = np.argsort(predictions[0])[-5:][::-1]
                        for idx in top5_idx:
                            prob = float(predictions[0][idx])
                            st.progress(prob, text=f"{CLASS_NAMES[idx]}: {prob:.2%}")
                        
                    except Exception as e:
                        st.error(f"Lỗi: {str(e)}")
                        st.info("💡 Hãy đảm bảo ảnh upload có định dạng đúng (JPG, PNG)")

else:
    st.info("👈 Vui lòng upload file model.onnx ở thanh bên trái")
    st.markdown("""
    ### Hướng dẫn:
    1. Upload file `model.onnx` đã chuyển đổi
    2. Upload ảnh món ăn
    3. Nhấn "Dự đoán"
    """)

st.markdown("---")
st.caption("🍜 Model yêu cầu input: 224x224x3 (ảnh RGB)")

