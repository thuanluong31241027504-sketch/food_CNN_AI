import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="Vietnamese Food CNN", page_icon="🍜")

st.title("🍜 Nhận diện món ăn Việt Nam")

# 30 món ăn
CLASS_NAMES = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh cuon', 'Banh hoi',
    'Banh uot', 'Banh xeo', 'Bo bia', 'Bun bo Hue', 'Bun cha',
    'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Cao lau', 'Com ga',
    'Chao long', 'Cha gio', 'Cha ram', 'Com tam', 'Goi cuon',
    'Hu tieu', 'Mi Quang', 'Pho', 'Xoi', 'Cha ca La Vong',
    'Lau', 'Bo kho', 'Ga nuong', 'Vit quay', 'Heo quay'
]

# ========== PHẦN 1: UPLOAD MODEL ==========
st.header("📁 Bước 1: Upload model")

uploaded_model = st.file_uploader("Chọn file model.onnx", type=['onnx'])

session = None

if uploaded_model:
    # Lưu model tạm thời
    with open("temp_model.onnx", "wb") as f:
        f.write(uploaded_model.getbuffer())
    
    # Load model
    session = ort.InferenceSession("temp_model.onnx")
    st.success("✅ Model đã được load thành công!")
    
    # Hiển thị thông tin model
    input_info = session.get_inputs()[0]
    st.info(f"📐 Model yêu cầu input: {input_info.shape}")

# ========== PHẦN 2: UPLOAD ẢNH ==========
st.header("📸 Bước 2: Upload ảnh")

uploaded_image = st.file_uploader("Chọn ảnh món ăn...", type=['jpg', 'jpeg', 'png'])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Ảnh của bạn", width=300)

# ========== PHẦN 3: NÚT DỰ ĐOÁN ==========
st.header("🎯 Bước 3: Dự đoán")

# NÚT DỰ ĐOÁN - luôn hiển thị
if st.button("🔍 DỰ ĐOÁN NGAY", type="primary", use_container_width=True):
    
    # Kiểm tra đã upload model chưa
    if session is None:
        st.error("❌ Vui lòng upload model ONNX trước!")
    # Kiểm tra đã upload ảnh chưa
    elif uploaded_image is None:
        st.error("❌ Vui lòng upload ảnh trước!")
    else:
        with st.spinner("Đang xử lý..."):
            try:
                # Xử lý ảnh
                img = image.copy()
                
                # Chuyển sang RGB nếu cần
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                
                # Resize (lấy từ thông tin model)
                input_shape = session.get_inputs()[0].shape
                target_size = (input_shape[1], input_shape[2])  # (224, 224)
                
                img = img.resize(target_size)
                
                # Chuyển sang array
                img_array = np.array(img).astype(np.float32) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Dự đoán
                input_name = session.get_inputs()[0].name
                predictions = session.run(None, {input_name: img_array})[0]
                
                idx = np.argmax(predictions[0])
                confidence = float(predictions[0][idx])
                
                st.success(f"### 🎯 Kết quả: **{CLASS_NAMES[idx]}**")
                st.info(f"📊 Độ tin cậy: **{confidence:.2%}**")
                
                # Top 5
                st.write("**Top 5 dự đoán:**")
                top5_idx = np.argsort(predictions[0])[-5:][::-1]
                for i, idx in enumerate(top5_idx, 1):
                    prob = float(predictions[0][idx])
                    st.write(f"{i}. **{CLASS_NAMES[idx]}** - {prob:.2%}")
                    st.progress(prob)
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

st.markdown("---")
st.caption("💡 Hướng dẫn: Upload model → Upload ảnh → Bấm nút Dự đoán")

