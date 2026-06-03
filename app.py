import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os

st.set_page_config(page_title="Check Model ONNX", page_icon="🔍", layout="wide")

st.title("🔍 Kiểm tra Model ONNX")
st.markdown("---")

# 30 món ăn từ dataset
CLASS_NAMES = [
    'Banh beo', 'Banh bot loc', 'Banh can', 'Banh cuon', 'Banh hoi',
    'Banh uot', 'Banh xeo', 'Bo bia', 'Bun bo Hue', 'Bun cha',
    'Bun mam', 'Bun rieu', 'Bun thit nuong', 'Cao lau', 'Com ga',
    'Chao long', 'Cha gio', 'Cha ram', 'Com tam', 'Goi cuon',
    'Hu tieu', 'Mi Quang', 'Pho', 'Xoi', 'Cha ca La Vong',
    'Lau', 'Bo kho', 'Ga nuong', 'Vit quay', 'Heo quay'
]

# Session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model_info' not in st.session_state:
    st.session_state.model_info = None
if 'session' not in st.session_state:
    st.session_state.session = None

# Sidebar - Upload model
with st.sidebar:
    st.header("📁 Upload Model")
    
    uploaded_model = st.file_uploader("Chọn file model.onnx", type=['onnx'])
    
    if uploaded_model:
        with open("model.onnx", "wb") as f:
            f.write(uploaded_model.getbuffer())
        st.success("✅ Đã upload model!")
        
        # Load lại model
        try:
            session = ort.InferenceSession("model.onnx")
            st.session_state.session = session
            st.session_state.model_loaded = True
            
            # Lấy thông tin model
            info = {}
            info['inputs'] = []
            for inp in session.get_inputs():
                info['inputs'].append({
                    'name': inp.name,
                    'shape': inp.shape,
                    'type': inp.type
                })
            
            info['outputs'] = []
            for out in session.get_outputs():
                info['outputs'].append({
                    'name': out.name,
                    'shape': out.shape,
                    'type': out.type
                })
            
            st.session_state.model_info = info
            st.rerun()
            
        except Exception as e:
            st.error(f"Lỗi load model: {str(e)}")
            st.session_state.model_loaded = False

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Thông tin Model")
    
    if st.session_state.model_loaded and st.session_state.model_info:
        info = st.session_state.model_info
        
        st.subheader("📥 Input:")
        for inp in info['inputs']:
            st.write(f"- **Name:** `{inp['name']}`")
            st.write(f"- **Shape:** `{inp['shape']}`")
            st.write(f"- **Type:** `{inp['type']}`")
            
            # Gợi ý kích thước ảnh
            if len(inp['shape']) == 4:
                height = inp['shape'][1]
                width = inp['shape'][2]
                channels = inp['shape'][3]
                st.info(f"💡 **Ảnh cần resize về:** `{height} x {width} x {channels}`")
        
        st.subheader("📤 Output:")
        for out in info['outputs']:
            st.write(f"- **Name:** `{out['name']}`")
            st.write(f"- **Shape:** `{out['shape']}`")
            st.write(f"- **Type:** `{out['type']}`")
            
            if len(out['shape']) == 2:
                num_classes = out['shape'][1]
                st.info(f"💡 **Số classes:** `{num_classes}` món ăn")
                
                if num_classes == 30:
                    st.success("✅ Model có 30 classes - khớp với dataset!")
                else:
                    st.warning(f"⚠️ Model có {num_classes} classes, nhưng danh sách hiện tại có 30 classes")
        
        st.balloons()
        st.success("✅ Model loaded thành công!")
        
    else:
        st.info("👈 Chưa có model. Hãy upload file .onnx ở thanh bên trái")

with col2:
    st.header("🔮 Test Dự đoán")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Vui lòng upload model trước!")
    else:
        uploaded_image = st.file_uploader("Chọn ảnh để test...", type=['jpg', 'jpeg', 'png', 'webp'])
        
        if uploaded_image:
            # Đọc ảnh
            image = Image.open(uploaded_image)
            
            st.image(image, caption="Ảnh gốc", width=250)
            
            # Hiển thị thông tin ảnh gốc
            st.write(f"**Kích thước gốc:** {image.size}")
            st.write(f"**Mode:** {image.mode}")
            
            # Lấy thông tin input shape từ model
            if st.session_state.model_info:
                input_shape = st.session_state.model_info['inputs'][0]['shape']
                
                if len(input_shape) == 4:
                    target_size = (input_shape[1], input_shape[2])  # (height, width)
                    expected_channels = input_shape[3]
                    
                    st.info(f"📐 Model yêu cầu: {target_size[0]}x{target_size[1]}, {expected_channels} channels")
                    
                    # Xử lý ảnh
                    with st.spinner("Đang xử lý..."):
                        # Chuyển sang RGB nếu cần
                        if image.mode == 'RGBA':
                            image = image.convert('RGB')
                            st.write("✅ Đã chuyển RGBA → RGB")
                        elif image.mode != 'RGB' and expected_channels == 3:
                            image = image.convert('RGB')
                            st.write(f"✅ Đã chuyển {image.mode} → RGB")
                        
                        # Resize
                        img = image.resize(target_size)
                        st.write(f"✅ Đã resize: {img.size}")
                        
                        # Chuyển sang array
                        img_array = np.array(img).astype(np.float32)
                        
                        # Normalize
                        img_array = img_array / 255.0
                        
                        # Reshape
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        st.write(f"✅ Final shape: {img_array.shape}")
                    
                    if st.button("🚀 Dự đoán", type="primary", use_container_width=True):
                        with st.spinner("Đang dự đoán..."):
                            try:
                                session = st.session_state.session
                                input_name = session.get_inputs()[0].name
                                
                                predictions = session.run(None, {input_name: img_array})[0]
                                
                                predicted_idx = np.argmax(predictions[0])
                                confidence = float(np.max(predictions[0]))
                                
                                st.success(f"### 🎯 Kết quả: **{CLASS_NAMES[predicted_idx]}**")
                                st.info(f"📊 Độ tin cậy: **{confidence:.2%}**")
                                
                                # Hiển thị Top 5
                                st.write("### 🏆 Top 5 dự đoán:")
                                top5_idx = np.argsort(predictions[0])[-5:][::-1]
                                for idx in top5_idx:
                                    prob = float(predictions[0][idx])
                                    st.progress(prob, text=f"{CLASS_NAMES[idx]}: {prob:.2%}")
                                
                                # Hiển thị tất cả scores (bar chart)
                                st.write("### 📊 Chi tiết tất cả classes:")
                                
                                # Chỉ hiển thị top 10 để dễ nhìn
                                top10_idx = np.argsort(predictions[0])[-10:][::-1]
                                
                                import matplotlib.pyplot as plt
                                fig, ax = plt.subplots(figsize=(8, 4))
                                names = [CLASS_NAMES[i] for i in top10_idx]
                                scores = [predictions[0][i] for i in top10_idx]
                                
                                bars = ax.barh(range(10), scores, color='coral')
                                ax.set_yticks(range(10))
                                ax.set_yticklabels(names, fontsize=9)
                                ax.set_xlabel('Confidence')
                                ax.set_title('Top 10 predictions')
                                ax.set_xlim(0, 1)
                                
                                for i, (bar, score) in enumerate(zip(bars, scores)):
                                    ax.text(score + 0.02, i, f'{score:.2%}', va='center', fontsize=8)
                                
                                st.pyplot(fig)
                                
                            except Exception as e:
                                st.error(f"Lỗi khi dự đoán: {str(e)}")

# Footer
st.markdown("---")
st.caption("🔧 App kiểm tra model ONNX - Hiển thị thông tin input/output và test dự đoán")

