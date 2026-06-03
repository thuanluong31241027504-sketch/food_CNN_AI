import streamlit as st
import os
import random
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tempfile

# Cấu hình trang
st.set_page_config(page_title="Vietnamese Food CNN", page_icon="🍜", layout="wide")

st.title("🍜 Vietnamese Food Recognition with CNN")
st.markdown("---")

# ==================== THAM SỐ GIỐNG CODE GỐC ====================
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
IMAGE_SIZE = (128, 128)

# ==================== ĐỊNH NGHĨA MODEL CNN ====================
class VietnameseFoodCNN(nn.Module):
    def __init__(self, num_classes=30):
        super(VietnameseFoodCNN, self).__init__()
        
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.30)
        
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.dropout3 = nn.Dropout(0.35)
        
        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2)
        self.dropout4 = nn.Dropout(0.40)
        
        # Head
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Block 4
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = self.pool4(x)
        x = self.dropout4(x)
        
        # Head
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        
        return x

# ==================== DATASET CLASS ====================
class VietnameseFoodDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.classes = sorted(dataframe['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['filepath']
        label = self.dataframe.iloc[idx]['label']
        
        image = Image.open(img_path).convert('RGB')
        label_idx = self.class_to_idx[label]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx

# ==================== HÀM TẠO DATAFRAME ====================
def create_dataframe(directory):
    filepaths, labels = [], []
    for label in os.listdir(directory):
        class_dir = os.path.join(directory, label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    filepaths.append(os.path.join(class_dir, file))
                    labels.append(label)
    return pd.DataFrame({'filepath': filepaths, 'label': labels})

# ==================== DATA AUGMENTATION ====================
train_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=20),
    transforms.ToTensor(),
])

valid_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
])

# Session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'classes' not in st.session_state:
    st.session_state.classes = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'data_ready' not in st.session_state:
    st.session_state.data_ready = False

# Sidebar
with st.sidebar:
    st.header("⚙️ Thông số (giống code gốc)")
    st.write(f"Batch Size: {BATCH_SIZE}")
    st.write(f"Learning Rate: {LEARNING_RATE}")
    st.write(f"Epochs: {EPOCHS}")
    st.write(f"Image Size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")

# Main content
tab1, tab2, tab3 = st.tabs(["📁 Upload & Load Data", "🏗️ Build & Train", "🔮 Predict"])

# ==================== TAB 1: UPLOAD DATA ====================
with tab1:
    st.header("1. Upload và chuẩn bị dữ liệu")
    
    st.info("""
    📌 **Hướng dẫn tải dataset:**
    1. Truy cập: https://www.kaggle.com/datasets/quandang/vietnamese-foods
    2. Click "Download" để tải file ZIP
    3. Upload file ZIP lên đây
    """)
    
    uploaded_zip = st.file_uploader("Chọn file ZIP dataset", type=['zip'])
    
    if uploaded_zip:
        if st.button("📂 Giải nén và load dữ liệu", type="primary", use_container_width=True):
            with st.spinner("Đang giải nén và xử lý dữ liệu..."):
                try:
                    # Giải nén
                    temp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(temp_dir, "dataset.zip")
                    
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_zip.getbuffer())
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Tìm thư mục Images
                    images_path = None
                    for root, dirs, files in os.walk(temp_dir):
                        if 'Images' in dirs:
                            images_path = os.path.join(root, 'Images')
                            break
                        if 'Train' in dirs:
                            images_path = root
                            break
                    
                    if images_path is None:
                        images_path = temp_dir
                    
                    # Load dataframes
                    train_df = create_dataframe(os.path.join(images_path, 'Train'))
                    valid_df = create_dataframe(os.path.join(images_path, 'Validate'))
                    test_df = create_dataframe(os.path.join(images_path, 'Test'))
                    
                    if len(train_df) == 0:
                        st.error("❌ Không tìm thấy dữ liệu! Hãy chắc chắn file ZIP có cấu trúc đúng.")
                    else:
                        st.write(f"📊 **Train:** {len(train_df)} | **Validate:** {len(valid_df)} | **Test:** {len(test_df)}")
                        
                        # Hiển thị 5 ảnh mẫu
                        fig, axes = plt.subplots(1, min(5, len(train_df)), figsize=(15, 4))
                        if len(train_df) == 1:
                            axes = [axes]
                        for i in range(min(5, len(train_df))):
                            idx = random.randint(0, len(train_df) - 1)
                            img_path = train_df.iloc[idx]['filepath']
                            label = train_df.iloc[idx]['label']
                            img = mpimg.imread(img_path)
                            axes[i].imshow(img)
                            axes[i].set_title(label, fontsize=8)
                            axes[i].axis('off')
                        st.pyplot(fig)
                        
                        # Tạo datasets
                        train_dataset = VietnameseFoodDataset(train_df, transform=train_transform)
                        valid_dataset = VietnameseFoodDataset(valid_df, transform=valid_transform)
                        
                        st.session_state.train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                        st.session_state.valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
                        st.session_state.classes = train_dataset.classes
                        st.session_state.data_ready = True
                        st.session_state.data_path = temp_dir
                        
                        st.success(f"✅ Load thành công! {len(st.session_state.classes)} classes")
                        st.write(f"**Classes:** {', '.join(st.session_state.classes[:10])}...")
                        
                except Exception as e:
                    st.error(f"Lỗi: {str(e)}")

# ==================== TAB 2: TRAIN ====================
with tab2:
    st.header("2. Xây dựng và Train CNN Model")
    
    if not st.session_state.get('data_ready'):
        st.warning("⚠️ Vui lòng upload và load dữ liệu ở tab 'Upload & Load Data' trước!")
    else:
        st.success(f"✅ Dữ liệu đã sẵn sàng - {len(st.session_state.classes)} classes")
        
        if st.button("🚀 BẮT ĐẦU TRAINING", type="primary", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                num_classes = len(st.session_state.classes)
                
                model = VietnameseFoodCNN(num_classes=num_classes).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
                
                history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
                
                for epoch in range(EPOCHS):
                    # Training
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for images, labels in st.session_state.train_loader:
                        images, labels = images.to(device), labels.to(device)
                        
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    
                    train_acc = 100 * correct / total
                    train_loss = running_loss / len(st.session_state.train_loader)
                    history['accuracy'].append(train_acc / 100)
                    history['loss'].append(train_loss)
                    
                    # Validation
                    model.eval()
                    val_correct = 0
                    val_total = 0
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for images, labels in st.session_state.valid_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            val_loss += loss.item()
                            _, predicted = torch.max(outputs, 1)
                            val_total += labels.size(0)
                            val_correct += (predicted == labels).sum().item()
                    
                    val_acc = 100 * val_correct / val_total
                    val_loss = val_loss / len(st.session_state.valid_loader)
                    history['val_accuracy'].append(val_acc / 100)
                    history['val_loss'].append(val_loss)
                    
                    progress_bar.progress((epoch + 1) / EPOCHS)
                    status_text.text(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
                
                st.session_state.model = model
                st.session_state.trained = True
                st.session_state.history = history
                
                progress_bar.progress(100)
                status_text.text("✅ Training hoàn tất!")
                
                col1, col2 = st.columns(2)
                col1.metric("Training Accuracy", f"{history['accuracy'][-1]:.2%}")
                col2.metric("Validation Accuracy", f"{history['val_accuracy'][-1]:.2%}")
                
                # Vẽ biểu đồ
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                ax1.plot(history['accuracy'], label='Train', marker='o')
                ax1.plot(history['val_accuracy'], label='Validation', marker='o')
                ax1.set_title('Model Accuracy')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Accuracy')
                ax1.legend()
                ax1.grid(True)
                
                ax2.plot(history['loss'], label='Train Loss', marker='o')
                ax2.plot(history['val_loss'], label='Validation Loss', marker='o')
                ax2.set_title('Model Loss')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.legend()
                ax2.grid(True)
                
                st.pyplot(fig)
                
                st.balloons()
                st.success("🎉 Training thành công!")
                
            except Exception as e:
                st.error(f"Lỗi: {str(e)}")

# ==================== TAB 3: PREDICT ====================
with tab3:
    st.header("3. Dự đoán món ăn từ ảnh")
    
    if not st.session_state.get('trained'):
        st.warning("⚠️ Vui lòng train model ở tab 'Build & Train' trước!")
    else:
        uploaded_file = st.file_uploader("Chọn ảnh món ăn", type=['jpg', 'jpeg', 'png', 'webp'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh đã upload", width=250)
            
            predict_transform = transforms.Compose([
                transforms.Resize(IMAGE_SIZE),
                transforms.ToTensor(),
            ])
            
            if st.button("🔍 Dự đoán", type="primary"):
                with st.spinner("Đang xử lý..."):
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = st.session_state.model.to(device)
                    model.eval()
                    
                    img_tensor = predict_transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(img_tensor)
                        probabilities = F.softmax(outputs[0], dim=0)
                        prediction = torch.argmax(probabilities).item()
                        confidence = probabilities[prediction].item()
                    
                    food_name = st.session_state.classes[prediction]
                    
                    st.success(f"### 🎯 Kết quả: **{food_name}**")
                    st.info(f"📊 Độ tin cậy: **{confidence:.2%}**")
                    
                    st.write("### Top 5 dự đoán:")
                    top5 = torch.topk(probabilities, min(5, len(probabilities)))
                    for i in range(top5.indices.size(0)):
                        idx = top5.indices[i].item()
                        prob = top5.values[i].item()
                        st.progress(prob, text=f"{st.session_state.classes[idx]}: {prob:.2%}")

st.markdown("---")
st.caption("🍜 CNN Model - Nhận diện món ăn Việt Nam | Upload dataset từ Kaggle")

