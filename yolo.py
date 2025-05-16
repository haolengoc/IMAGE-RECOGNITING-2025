import os
from ultralytics import YOLO

# Đường dẫn đến dataset
data_path = r'D:\ueh\yolo_dataset'  # Thay bằng đường dẫn thực tế nếu cần

# Kiểm tra GPU (tùy chọn, chỉ để xác nhận)
try:
    import torch
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("PyTorch không được cài đặt, sẽ chạy trên CPU.")


# Khởi tạo mô hình YOLOv8
model = YOLO('yolov8n.pt')  # Sử dụng mô hình YOLOv8 nano pre-trained

# Huấn luyện mô hình
model.train(
    data=r'D:\ueh\yolo_dataset\data.yaml',  # Đường dẫn đến file data.yaml
    epochs=50,                      # Số epoch
    imgsz=420,                      # Kích thước ảnh 420x420
    batch=8,                    
    device=0 if torch.cuda.is_available() else 'cpu',  # Sử dụng GPU nếu có, nếu không dùng CPU
    project='yolo_results',         # Thư mục lưu kết quả
)

print(f"Huấn luyện hoàn tất! Kết quả được lưu tại: {os.path.abspath('yolo_results/yolo_food_detection')}")
