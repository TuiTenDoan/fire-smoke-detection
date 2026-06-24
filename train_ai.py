from ultralytics import YOLO

if __name__ == '__main__':
    print("🚒 Đang khởi động Hệ thống Cảnh báo Cháy sớm...")
    # Tải bộ não YOLOv8 bản Nano (cực nhẹ, cực nhanh)
    model = YOLO("yolov8n.pt") 
    
    print("🔥 Bắt đầu huấn luyện AI nhận diện Khói và Lửa...")
    results = model.train(
        data="F:\dataset_cuu_hoa\data.yaml", # Khai báo vị trí cuốn sách giáo khoa
        epochs=50,       # Cho học 50 vòng
        imgsz=640,       # Cắt ảnh vuông vức 640x640
        batch=8,         # 1650 Ti gánh 8 ảnh/lượt là đẹp nhất
        device=0,        # Gọi hồn card rời làm việc
        workers=2,       
        name="fire_smoke_model" # Tên thư mục tốt nghiệp
    )
    
    print("✅ Huấn luyện hoàn tất! Bộ não cứu hỏa đã sẵn sàng hoạt động.")