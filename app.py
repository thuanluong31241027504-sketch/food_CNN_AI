import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
import tempfile
import time

# Cấu hình trang
st.set_page_config(
    page_title="Train CNN - Vietnamese Food", 
    page_icon="🍜",
    layout="wide"
)

# Title
st.title("🍜 Train CNN Model Trực Tiếp trên Streamlit")
st.markdown("---")

# Khởi tạo session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'classes' not in st.session_state:
    st.session_state.classes = None
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False

# Sidebar - Parameters
with st.sidebar:
    st.header("⚙️ Cài đặt Model")
    
    # Model parameters
    st.subheader("🏗️ Kiến trúc CNN")
    conv_layers = st.select_slider("Số Conv Layers", options=[1, 2, 3], value=2)
    filters_start = st.selectbox("Số filters khởi đầu", [16, 32, 64], index=1)
    dense_units = st.selectbox("Số neuron Dense layer", [64, 128, 256], index=1)
    dropout_rate = st.slider("Dropout rate", 0.2, 0.7, 0.5, 0.1)
    
    # Training parameters
    st.subheader("🎯 Training Parameters")
    epochs = st.slider("Số epochs", 1, 15, 5)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32], value=16)
    learning_rate = st.number_input("Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    st.markdown("---")
    st.info("💡 **Mẹo:** Bắt đầu với epochs=5 để test trước")

# Main content - 2 cột
col1, col2 = st.columns([1, 1])

# ============ CỘT 1: CHUẨN BỊ DỮ LIỆU ============
with col1:
    st.header("📁 1. Chuẩn bị dữ liệu")
    
    # Option 1: Dùng dữ liệu mẫu (tự tạo)
    use_sample = st.checkbox("✅ Dùng dữ liệu mẫu (khuyến nghị)", value=True)
    
    if use_sample:
        st.info("Dữ liệu mẫu sẽ tự động tạo ảnh giả để test training")
        
        if st.button("🔄 Tạo dữ liệu mẫu", type="primary", use_container_width=True):
            with st.spinner("Đang tạo dữ liệu mẫu..."):
                # Tạo thư mục tạm
                temp_dir = tempfile.mkdtemp()
                
                # Tên các classes
                classes = ['Pho', 'Bun_Cha', 'Banh_Mi', 'Com_Tam', 'Banh_Xeo']
                st.session_state.classes = classes
                
                # Tạo ảnh giả
                from PIL import ImageDraw
                
                for class_name in classes:
                    # Tạo thư mục Train
                    train_dir = os.path.join(temp_dir, 'Train', class_name)
                    os.makedirs(train_dir, exist_ok=True)
                    
                    # Tạo thư mục Validation
                    val_dir = os.path.join(temp_dir, 'Validation', class_name)
                    os.makedirs(val_dir, exist_ok=True)
                    
                    # Tạo 30 ảnh train, 10 ảnh validation mỗi class
                    for i in range(30):
                        img = Image.new('RGB', (128, 128), color=(
                            np.random.randint(100, 255),
                            np.random.randint(100, 255),
                            np.random.randint(100, 255)
                        ))
                        draw = ImageDraw.Draw(img)
                        draw.ellipse([20, 20, 108, 108], fill=(
                            np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)
                        ))
                        img.save(os.path.join(train_dir, f'{class_name}_{i}.jpg'))
                    
                    for i in range(10):
                        img = Image.new('RGB', (128, 128), color=(
                            np.random.randint(100, 255),
                            np.random.randint(100, 255),
                            np.random.randint(100, 255)
                        ))
                        draw = ImageDraw.Draw(img)
                        draw.ellipse([20, 20, 108, 108], fill=(
                            np.random.randint(0, 255),
                            np.random.randint(0, 255),
                            np.random.randint(0, 255)
                        ))
                        img.save(os.path.join(val_dir, f'{class_name}_val_{i}.jpg'))
                
                st.session_state.data_path = temp_dir
                st.session_state.data_ready = True
                
                st.success(f"✅ Đã tạo xong dữ liệu mẫu!")
                st.write(f"📊 **{len(classes)} classes:** {', '.join(classes)}")
                st.write(f"🖼️ **30 ảnh train + 10 ảnh validation** mỗi class")
    
    # Option 2: Upload dữ liệu của bạn
    else:
        st.warning("⚠️ Upload file ZIP có cấu trúc:")
        st.code("""
data.zip
├── Train/
│   ├── Pho/
│   ├── BunCha/
│   └── ...
└── Validation/
    ├── Pho/
    ├── BunCha/
    └── ...
        """)
        
        uploaded_zip = st.file_uploader("Chọn file ZIP", type=['zip'])
        
        if uploaded_zip and st.button("📦 Giải nén dữ liệu", use_container_width=True):
            with st.spinner("Đang giải nén..."):
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, "data.zip")
                
                with open(zip_path, "wb") as f:
                    f.write(uploaded_zip.getbuffer())
                
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
                
                # Tìm thư mục Train và Validation
                train_path = None
                val_path = None
                
                for root, dirs, files in os.walk(temp_dir):
                    if 'Train' in dirs:
                        train_path = os.path.join(root, 'Train')
                    if 'Validation' in dirs or 'Validate' in dirs:
                        val_path = os.path.join(root, 'Validation') if 'Validation' in dirs else os.path.join(root, 'Validate')
                
                if train_path and val_path:
                    # Lấy danh sách classes
                    classes = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
                    st.session_state.classes = classes
                    st.session_state.data_path = temp_dir
                    st.session_state.data_ready = True
                    st.success(f"✅ Dữ liệu đã sẵn sàng! {len(classes)} classes detected")
                else:
                    st.error("❌ Không tìm thấy thư mục Train/Validation!")

# ============ CỘT 2: XÂY DỰNG VÀ TRAIN MODEL ============
with col2:
    st.header("🏗️ 2. Xây dựng & Train Model")
    
    if not st.session_state.data_ready:
        st.info("👈 Hãy tạo hoặc upload dữ liệu ở cột bên trái trước")
    else:
        st.success(f"✅ Dữ liệu đã sẵn sàng - {len(st.session_state.classes)} classes")
        
        if st.button("🚀 BẮT ĐẦU TRAINING", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metric_text = st.empty()
            
            try:
                # 1. Load dữ liệu
                status_text.text("📂 Đang load dữ liệu...")
                
                train_path = os.path.join(st.session_state.data_path, 'Train')
                val_path = None
                
                # Tìm validation path
                for root, dirs, files in os.walk(st.session_state.data_path):
                    if 'Validation' in dirs:
                        val_path = os.path.join(root, 'Validation')
                        break
                    elif 'Validate' in dirs:
                        val_path = os.path.join(root, 'Validate')
                        break
                
                if val_path is None:
                    # Nếu không có validation, dùng 20% từ train
                    val_path = None
                    st.info("Không tìm thấy validation folder, sẽ tự động tách 20% từ train")
                
                # Data augmentation
                train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    validation_split=0.2 if val_path is None else 0.0
                )
                
                if val_path:
                    val_datagen = ImageDataGenerator(rescale=1./255)
                    
                    train_generator = train_datagen.flow_from_directory(
                        train_path,
                        target_size=(128, 128),
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=True
                    )
                    
                    val_generator = val_datagen.flow_from_directory(
                        val_path,
                        target_size=(128, 128),
                        batch_size=batch_size,
                        class_mode='categorical',
                        shuffle=False
                    )
                else:
                    train_generator = train_datagen.flow_from_directory(
                        train_path,
                        target_size=(128, 128),
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset='training',
                        shuffle=True
                    )
                    
                    val_generator = train_datagen.flow_from_directory(
                        train_path,
                        target_size=(128, 128),
                        batch_size=batch_size,
                        class_mode='categorical',
                        subset='validation',
                        shuffle=False
                    )
                
                num_classes = len(train_generator.class_indices)
                st.session_state.classes = list(train_generator.class_indices.keys())
                
                status_text.text(f"✅ Dữ liệu loaded: {train_generator.samples} train, {val_generator.samples} val")
                
                # 2. Xây dựng model
                status_text.text("🏗️ Đang xây dựng CNN model...")
                
                model = Sequential()
                model.add(Input(shape=(128, 128, 3)))
                
                # Conv layers
                filters = filters_start
                for i in range(conv_layers):
                    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
                    model.add(BatchNormalization())
                    model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
                    model.add(BatchNormalization())
                    model.add(MaxPooling2D(2, 2))
                    model.add(Dropout(min(0.25 + i*0.05, 0.5)))
                    filters *= 2
                
                # Head
                model.add(GlobalAveragePooling2D())
                model.add(Dense(dense_units, activation='relu'))
                model.add(Dropout(dropout_rate))
                model.add(Dense(num_classes, activation='softmax'))
                
                # Compile
                model.compile(
                    optimizer=Adam(learning_rate=learning_rate),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                # Hiển thị model summary
                with st.expander("📋 Xem cấu trúc model"):
                    summary_str = []
                    model.summary(print_fn=lambda x: summary_str.append(x))
                    st.code('\n'.join(summary_str))
                
                # 3. Training
                status_text.text("🎯 Đang training model...")
                
                # Callback cập nhật progress
                class MyCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        metric_text.text(f"Epoch {epoch+1}/{epochs} - Acc: {logs['accuracy']:.4f} - Val Acc: {logs['val_accuracy']:.4f}")
                
                history = model.fit(
                    train_generator,
                    epochs=epochs,
                    validation_data=val_generator,
                    callbacks=[MyCallback()],
                    verbose=0
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Training hoàn tất!")
                
                # Lưu model
                st.session_state.model = model
                st.session_state.history = history
                
                # Hiển thị kết quả
                final_acc = history.history['accuracy'][-1]
                final_val_acc = history.history['val_accuracy'][-1]
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Training Accuracy", f"{final_acc:.2%}")
                with col_b:
                    st.metric("Validation Accuracy", f"{final_val_acc:.2%}")
                
                # Vẽ biểu đồ
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(history.history['accuracy'], label='Train', marker='o')
                ax1.plot(history.history['val_accuracy'], label='Validation', marker='o')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                ax1.grid(True)
                
                ax2.plot(history.history['loss'], label='Train', marker='o')
                ax2.plot(history.history['val_loss'], label='Validation', marker='o')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig)
                
                # Lưu model
                model_path = os.path.join(tempfile.gettempdir(), 'food_model.h5')
                model.save(model_path)
                
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="💾 Tải model về máy (model.h5)",
                        data=f,
                        file_name="vietnamese_food_model.h5",
                        mime="application/octet-stream"
                    )
                
                st.balloons()
                st.success("🎉 Training thành công! Bạn có thể dùng model để dự đoán bên dưới.")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.info("💡 Thử giảm epochs xuống 3-5 hoặc batch_size xuống 8 để test trước")

# ============ PHẦN 3: DỰ ĐOÁN ============
if st.session_state.model is not None:
    st.markdown("---")
    st.header("🔮 3. Dự đoán với model vừa train")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📤 Chọn ảnh món ăn", type=['jpg', 'jpeg', 'png'], key="predict")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh của bạn", width=300)
    
    with col2:
        if uploaded_file and st.button("🔍 Dự đoán", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý..."):
                # Preprocess
                img = image.resize((128, 128))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                predictions = st.session_state.model.predict(img_array)
                predicted_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                
                st.success(f"### 🎯 Kết quả: **{st.session_state.classes[predicted_idx]}**")
                st.info(f"### 📊 Độ tin cậy: **{confidence:.2%}**")
                
                # Hiển thị top 3
                st.write("### Top 3 dự đoán:")
                top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                for idx in top_3_idx:
                    st.progress(predictions[0][idx], text=f"{st.session_state.classes[idx]}: {predictions[0][idx]:.2%}")

st.markdown("---")
st.caption("💡 **Hướng dẫn:** Chọn 'Dùng dữ liệu mẫu' → Tạo dữ liệu mẫu → Bắt đầu training → Dự đoán thử")
