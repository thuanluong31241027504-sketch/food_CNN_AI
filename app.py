from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
import os
import pandas as pd
from google.colab import files

# 1. Upload ảnh cần test
print("📤 Upload ảnh cần test:")
uploaded_img = files.upload()
img_path = list(uploaded_img.keys())[0]

# 2. Upload model .h5
print("\n📤 Upload model .h5:")
uploaded_model = files.upload()
model_file = list(uploaded_model.keys())[0]

# 3. Load model
model = load_model(model_file)
print(f"\n✅ Model loaded: {model_file}")

# 4. Lấy class names từ model
# Thử lấy từ model trước
if hasattr(model, 'classes'):
    CLASS_NAMES = model.classes
    print(f"Classes từ model: {len(CLASS_NAMES)} classes")
else:
    # Hoặc lấy từ dataset
    print("📥 Download dataset để lấy class names...")
    path = kagglehub.dataset_download("quandang/vietnamese-foods")
    
    # Tìm thư mục Images
    images_path = None
    for root, dirs, files in os.walk(path):
        if 'Images' in dirs:
            images_path = os.path.join(root, 'Images')
            break
    
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
    
    train_df = create_dataframe(os.path.join(images_path, 'Train'))
    CLASS_NAMES = sorted(train_df['label'].unique())

print(f"\n📋 Danh sách {len(CLASS_NAMES)} món ăn:")
for i, name in enumerate(CLASS_NAMES[:10]):
    print(f"   {i+1}. {name}")
print("   ...")

# 5. Test với nhiều kích thước input khác nhau
print("\n" + "="*60)
print("🔍 TEST VỚI CÁC KÍCH THƯỚC KHÁC NHAU")
print("="*60)

# Đọc ảnh gốc
img_original = load_img(img_path)
print(f"\n📸 Ảnh gốc: {img_original.size}, mode: {img_original.mode}")

# Test với các target_size khác nhau
test_sizes = [(128, 128), (224, 224), (299, 299)]
results = {}

for target_size in test_sizes:
    print(f"\n--- Test với target_size = {target_size} ---")
    
    # Load ảnh với kích thước khác nhau
    img = load_img(img_path, target_size=target_size)
    
    # Hiển thị ảnh đã resize
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title(f"Resize {target_size}")
    plt.axis('off')
    
    # Tiền xử lý
    img_array = np.array(img)
    print(f"   Shape sau resize: {img_array.shape}")
    
    # Normalize
    img_array = img_array / 255.0
    print(f"   Min/Max: {img_array.min():.2f}/{img_array.max():.2f}")
    
    # Reshape
    img_array = img_array.reshape(1, target_size[0], target_size[1], 3)
    print(f"   Input shape: {img_array.shape}")
    
    # Dự đoán
    predictions = model.predict(img_array, verbose=0)
    predicted_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    food_name = CLASS_NAMES[predicted_idx]
    
    results[target_size] = (food_name, confidence, predictions[0])
    
    print(f"   🎯 Kết quả: {food_name}")
    print(f"   📊 Độ tin cậy: {confidence:.2%}")
    
    # Hiển thị Top 3
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    print(f"   🏆 Top 3:")
    for idx in top3_idx:
        print(f"      - {CLASS_NAMES[idx]}: {predictions[0][idx]:.2%}")
    
    # Hiển thị bar chart
    plt.subplot(1, 3, 2)
    top5_idx = np.argsort(predictions[0])[-5:][::-1]
    top5_names = [CLASS_NAMES[i] for i in top5_idx]
    top5_scores = [predictions[0][i] for i in top5_idx]
    plt.barh(range(5), top5_scores, color='coral')
    plt.yticks(range(5), top5_names)
    plt.xlabel('Confidence')
    plt.title(f'Top 5 - Size {target_size[0]}')
    plt.xlim(0, 1)
    
    plt.subplot(1, 3, 3)
    plt.bar(range(10), predictions[0][:10], color='skyblue')
    plt.xticks(range(10), CLASS_NAMES[:10], rotation=45, ha='right', fontsize=8)
    plt.title('Top 10 classes')
    plt.tight_layout()
    plt.show()

# 6. Kết luận
print("\n" + "="*60)
print("📊 KẾT LUẬN")
print("="*60)
print("\nKết quả dự đoán theo từng kích thước input:")
for size, (food, conf, _) in results.items():
    print(f"   {size[0]}x{size[1]}: {food} ({conf:.2%})")

# 7. Kiểm tra thông tin model
print("\n" + "="*60)
print("🔧 THÔNG TIN MODEL")
print("="*60)
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Number of layers: {len(model.layers)}")

