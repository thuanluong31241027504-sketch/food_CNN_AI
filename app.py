import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from torch.utils.data import DataLoader, Dataset
import zipfile

# Cấu hình trang
st.set_page_config(page_title="Vietnamese Food CNN", page_icon="🍜", layout="wide")

st.title("🍜 Vietnamese Food Recognition with CNN")
st.markdown("---")

# Khởi tạo session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'classes' not in st.session_state:
    st.session_state.classes = None
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False
if 'trained' not in st.session_state:
    st.session_state.trained = False

# ==================== ĐỊNH NGHĨA MODEL CNN ====================
# Đây chính là cấu trúc CNN bạn đã dùng trên Colab
class VietnameseFoodCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(VietnameseFoodCNN, self).__init__()
        
        # Block 1 (giống code gốc của bạn)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.30)
        )
        
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.35)
        )
        
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.40)
        )
        
        # Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ==================== PHẦN 1: CHUẨN BỊ DỮ LIỆU ====================
with st.sidebar:
    st.header("⚙️ Cài đặt")
    
    # Training parameters
    st.subheader("🎯 Training Parameters")
    epochs = st.slider("Số epochs", 1, 20, 5)
    batch_size = st.select_slider("Batch size", options=[8, 16, 32], value=16)
    learning_rate = st.number_input("Learning rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    st.markdown("---")
    st.info("💡 **Mẹo:** Bắt đầu với epochs=3-5 để test")

# Main content - 2 cột
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 1. Chuẩn bị dữ liệu")
    
    use_sample = st.checkbox("✅ Dùng dữ liệu mẫu (khuyến nghị)", value=True)
    
    if use_sample:
        st.info("Dữ liệu mẫu sẽ tự động tạo ảnh giả để test training")
        
        if st.button("🔄 Tạo dữ liệu mẫu", type="primary", use_container_width=True):
            with st.spinner("Đang tạo dữ liệu mẫu..."):
                temp_dir = tempfile.mkdtemp()
                
                # 10 món ăn Việt Nam phổ biến
                classes = ['Pho', 'Bun_Cha', 'Banh_Mi', 'Com_Tam', 'Banh_Xeo',
                          'Goi_Cuon', 'Bun_Bo_Hue', 'Cao_Lau', 'Mi_Quang', 'Cha_Ca']
                st.session_state.classes = classes
                
                from PIL import ImageDraw
                
                for class_name in classes:
                    train_dir = os.path.join(temp_dir, 'Train', class_name)
                    os.makedirs(train_dir, exist_ok=True)
                    
                    # Tạo 30 ảnh train mỗi class
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
                
                st.session_state.data_path = temp_dir
                st.session_state.data_ready = True
                
                st.success(f"✅ Đã tạo xong dữ liệu mẫu!")
                st.write(f"📊 **{len(classes)} classes:** {', '.join(classes[:5])}...")
                st.write(f"🖼️ **30 ảnh train** mỗi class")

with col2:
    st.header("🏗️ 2. Xây dựng & Train Model")
    
    if not st.session_state.data_ready:
        st.info("👈 Hãy tạo dữ liệu mẫu ở cột bên trái trước")
    else:
        st.success(f"✅ Dữ liệu đã sẵn sàng - {len(st.session_state.classes)} classes")
        
        if st.button("🚀 BẮT ĐẦU TRAINING", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            metric_text = st.empty()
            
            try:
                # 1. Load dữ liệu với PyTorch
                status_text.text("📂 Đang load dữ liệu...")
                
                from torchvision import datasets, transforms
                
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                train_path = os.path.join(st.session_state.data_path, 'Train')
                full_dataset = datasets.ImageFolder(train_path, transform=transform)
                
                # Chia train/val 80-20
                train_size = int(0.8 * len(full_dataset))
                val_size = len(full_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                num_classes = len(full_dataset.classes)
                st.session_state.classes = full_dataset.classes
                
                status_text.text(f"✅ Dữ liệu loaded: {train_size} train, {val_size} val")
                
                # 2. Khởi tạo model
                status_text.text("🏗️ Đang xây dựng CNN model...")
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = VietnameseFoodCNN(num_classes=num_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                
                # 3. Training
                status_text.text("🎯 Đang training model...")
                
                train_losses = []
                val_accs = []
                
                for epoch in range(epochs):
                    # Training
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    train_acc = 100 * correct / total
                    avg_loss = running_loss / len(train_loader)
                    train_losses.append(avg_loss)
                    
                    # Validation
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                    
                    val_acc = 100 * val_correct / val_total
                    val_accs.append(val_acc)
                    
                    # Update progress
                    progress_bar.progress((epoch + 1) / epochs)
                    metric_text.text(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
                
                progress_bar.progress(100)
                status_text.text("✅ Training hoàn tất!")
                
                # Lưu model
                st.session_state.model = model
                st.session_state.trained = True
                
                # Hiển thị kết quả
                final_train_acc = train_accs[-1] if train_accs else 0
                final_val_acc = val_accs[-1] if val_accs else 0
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Training Accuracy", f"{final_train_acc:.2f}%")
                with col_b:
                    st.metric("Validation Accuracy", f"{final_val_acc:.2f}%")
                
                # Vẽ biểu đồ
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                
                ax1.plot(train_losses, label='Train Loss', marker='o')
                ax1.set_title('Training Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True)
                
                ax2.plot(val_accs, label='Validation Accuracy', marker='o', color='green')
                ax2.set_title('Validation Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy (%)')
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig)
                
                # Lưu model để download
                model_path = os.path.join(tempfile.gettempdir(), 'food_model.pth')
                torch.save(model.state_dict(), model_path)
                
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="💾 Tải model về máy (PyTorch .pth)",
                        data=f,
                        file_name="vietnamese_food_cnn.pth",
                        mime="application/octet-stream"
                    )
                
                st.balloons()
                st.success("🎉 Training thành công! Bạn có thể dùng model để dự đoán bên dưới.")
                
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
                st.info("💡 Thử giảm epochs xuống 2-3 hoặc batch_size xuống 8 để test")

# ==================== PHẦN 3: DỰ ĐOÁN ====================
if st.session_state.trained and st.session_state.model is not None:
    st.markdown("---")
    st.header("🔮 3. Dự đoán với model vừa train")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📤 Chọn ảnh món ăn", type=['jpg', 'jpeg', 'png'], key="predict")
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh của bạn", width=250)
    
    with col2:
        if uploaded_file and st.button("🔍 Dự đoán", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý..."):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = st.session_state.model.to(device)
                model.eval()
                
                # Preprocess
                transform = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                img = transform(image).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(img)
                    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                    predicted_idx = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_idx].item()
                
                predicted_food = st.session_state.classes[predicted_idx]
                
                st.success(f"### 🎯 Kết quả: **{predicted_food}**")
                st.info(f"### 📊 Độ tin cậy: **{confidence:.2%}**")
                
                # Hiển thị top 5
                st.write("### 🏆 Top 5 dự đoán:")
                top5_idx = torch.argsort(probabilities, descending=True)[:5]
                for idx in top5_idx:
                    prob = probabilities[idx].item()
                    st.progress(prob, text=f"{st.session_state.classes[idx]}: {prob:.2%}")

st.markdown("---")
st.caption("💡 **Hướng dẫn:** Chọn 'Dùng dữ liệu mẫu' → Tạo dữ liệu mẫu → Bắt đầu training → Dự đoán thử")
