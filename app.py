import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="Vietnamese Food CNN", page_icon="🍜")

st.title("🍜 Nhận diện món ăn Việt Nam")

# ĐÚNG THỨ TỰ CLASSES TỪ DATASET
CLASS_NAMES = [
    'Banh beo',       # 0
    'Banh bot loc',   # 1
    'Banh can',       # 2
    'Banh canh',      # 3
    'Banh chung',     # 4
    'Banh cuon',      # 5
    'Banh duc',       # 6
    'Banh gio',       # 7
    'Banh khot',      # 8
    'Banh mi',        # 9
    'Banh pia',       # 10
    'Banh tet',       # 11
    'Banh trang nuong', # 12
    'Banh xeo',       # 13  ← Bánh xèo ở đây
    'Bun bo Hue',     # 14
    'Bun dau mam tom', # 15
    'Bun mam',        # 16
    'Bun rieu',       # 17
    'Bun thit nuong', # 18
    'Ca kho to',      # 19
    'Canh chua',      # 20
    'Cao lau',        # 21  ← Cao lầu ở đây
    'Chao long',      # 22
    'Com tam',        # 23
    'Goi cuon',       # 24
    'Hu tieu',        # 25
    'Mi quang',       # 26
    'Nem chua',       # 27
    'Pho',            # 28
    'Xoi xeo',        # 29
]

st.sidebar.header("📁 Upload Model")
uploaded_model = st.sidebar.file_uploader("Chọn file model.onnx", type=['onnx'])

session = None
if uploaded_model:
    with open("temp_model.onnx", "wb") as f:
        f.write(uploaded_model.getbuffer())
    session = ort.InferenceSession("temp_model.onnx")
    st.sidebar.success("✅ Model đã load thành công!")
    
    # Hiển thị thông tin model
    input_info = session.get_inputs()[0]
    st.sidebar.info(f"📐 Input shape: {input_info.shape}")

# Upload ảnh
st.header("📸 Chọn ảnh món ăn")
uploaded_image = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Ảnh của bạn", width=300)

# Nút dự đoán
if st.button("🔍 DỰ ĐOÁN", type="primary", use_container_width=True):
    
    if session is None:
        st.error("❌ Vui lòng upload model ONNX trước!")
    elif uploaded_image is None:
        st.error("❌ Vui lòng upload ảnh trước!")
    else:
        with st.spinner("Đang xử lý..."):
            try:
                # Xử lý ảnh
                img = image.copy()
                
                # Chuyển sang RGB nếu là RGBA
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Lấy kích thước yêu cầu từ model
                input_shape = session.get_inputs()[0].shape
                target_size = (input_shape[1], input_shape[2])
                
                # Resize
                img = img.resize(target_size)
                
                # Chuyển sang array và normalize
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Dự đoán
                input_name = session.get_inputs()[0].name
                predictions = session.run(None, {input_name: img_array})[0]
                
                idx = np.argmax(predictions[0])
                confidence = float(predictions[0][idx])
                
                # Hiển thị kết quả
                st.success(f"### 🎯 **{CLASS_NAMES[idx]}**")
                st.info(f"📊 Độ tin cậy: **{confidence:.2%}**")
                
                # Hiển thị Top 5
                st.write("---")
                st.write("### 🏆 Top 5 dự đoán:")
                top5_idx = np.argsort(predictions[0])[-5:][::-1]
                for i, idx in enumerate(top5_idx, 1):
                    prob = float(predictions[0][idx])
                    st.progress(prob, text=f"{i}. **{CLASS_NAMES[idx]}** - {prob:.2%}")
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

st.markdown("---")
st.caption("💡 Hướng dẫn: Upload model.onnx → Upload ảnh → Bấm Dự đoán")

