# Chạy trên Colab hoặc máy local
import onnxruntime as ort
import numpy as np

# Load model
try:
    session = ort.InferenceSession("model.onnx")
    print("✅ Model load thành công")
    
    # Kiểm tra input shape
    input_info = session.get_inputs()[0]
    print(f"Input name: {input_info.name}")
    print(f"Input shape: {input_info.shape}")
    
    # Kiểm tra output shape
    output_info = session.get_outputs()[0]
    print(f"Output shape: {output_info.shape}")
    
    # Test với dummy input
    dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    result = session.run(None, {input_info.name: dummy_input})
    print(f"✅ Test predict thành công, output shape: {result[0].shape}")
    
except Exception as e:
    print(f"❌ Lỗi: {e}")
