# README - Phân Loại Phương Tiện Giao Thông Đường Bộ

Tài liệu này mô tả đầy đủ dự án theo các câu hỏi:
- Dùng bộ dữ liệu nào
- Phân tích dữ liệu
- Đặc tính đặc trưng
- Tiền xử lý dữ liệu
- Dùng cách nào trong code để xử lý thông tin và dự đoán ảnh
- Cách thức hoạt động của phương pháp

## 1) Dùng bộ dữ liệu nào

Dự án sử dụng bộ dữ liệu ảnh phương tiện giao thông tải từ Kaggle, sau đó tổ chức lại theo cấu trúc thư mục phù hợp với bài toán phân loại ảnh:

```text
dataset/
├── train/
│   ├── Buses/
│   ├── Cars/
│   ├── Motobikes/
│   └── Trucks/
└── test/
    ├── Buses/
    ├── Cars/
    ├── Motobikes/
    └── Trucks/
```

Lưu ý:
- Dự án hỗ trợ cả tên lớp `Motobikes` và `Motorbikes` trong code để tương thích dữ liệu thực tế.
- Định dạng ảnh chính đang dùng là `.png`.

## 2) Phân tích dữ liệu

Số lượng ảnh thực dùng để train/test hiện tại:

### Train
- Buses: 204
- Cars: 813
- Motobikes: 1970
- Trucks: 268

### Test
- Buses: 50
- Cars: 203
- Motobikes: 492
- Trucks: 66

Tổng ảnh trong tập train + test đang dùng: 4066.

Nhận xét phân bố dữ liệu:
- Dữ liệu bị mất cân bằng rõ rệt, lớp `Motobikes` lớn hơn nhiều so với `Buses` và `Trucks`.
- Điều này dễ làm mô hình thiên lệch dự đoán về lớp chiếm ưu thế.
- Trong ảnh giao thông thực tế thường có nhiều phương tiện trong một khung hình, làm bài toán phân loại 1 nhãn/ảnh khó hơn.

## 3) Đặc tính đặc trưng của dữ liệu

Các đặc trưng chính mà mô hình học:
- Hình dạng tổng quát: chiều dài, chiều cao, khối thân xe.
- Cấu trúc cục bộ: bánh xe, kính, cabin, thùng xe.
- Màu sắc và kết cấu bề mặt (texture).
- Ngữ cảnh nền đường và mật độ giao thông.

Các trường hợp gây nhầm lẫn thường gặp:
- `Cars` và `Trucks` khi xe tải nhỏ hoặc góc nhìn xa.
- `Buses` và `Cars` khi xe bus ở xa, chiếm diện tích nhỏ.
- Ảnh có nhiều xe máy trong cùng khung hình dễ kéo xác suất về `Motobikes`.

## 4) Tiền xử lý dữ liệu (đã làm và đề xuất thêm)

### Các bước đã làm trong dự án
Trong [utils.py](utils.py):
- Resize ảnh về 224x224.
- Chuẩn hóa pixel về [0, 1].
- Chuyển kênh màu BGR -> RGB trước khi đưa vào model.
- Data augmentation cho train: xoay, zoom, lật ngang.
- Đọc ảnh đường dẫn Unicode trên Windows bằng fallback `np.fromfile + cv2.imdecode`.

Trong [train.py](train.py):
- Dùng class weights để giảm ảnh hưởng mất cân bằng lớp.
- Huấn luyện 2 giai đoạn:
  - Giai đoạn 1: train phần head phân loại.
  - Giai đoạn 2: fine-tune một phần VGG16 (từ `block5_conv1`).

### Đề xuất phương án tiền xử lý bổ sung
Nếu muốn tăng độ ổn định hơn nữa, có thể thử:
- Random brightness/contrast để mô phỏng điều kiện ánh sáng khác nhau.
- Random crop hoặc center crop để giảm nhiễu nền.
- Augmentation nâng cao: MixUp/CutMix.
- Dùng focal loss hoặc oversampling cho lớp hiếm.

## 5) Dùng cách nào trong code để xử lý thông tin và dự đoán ảnh

Dự án dùng Transfer Learning với VGG16 pretrained ImageNet.

Kiến trúc mô hình trong [utils.py](utils.py):
- Backbone: `VGG16(weights='imagenet', include_top=False)`.
- Head phân loại: `Flatten -> Dense(256, relu) -> Dropout -> Dense(4, softmax)`.

Quy trình huấn luyện trong [train.py](train.py):
- Tạo generator train/val/test từ thư mục.
- Train head với backbone đóng băng.
- Fine-tune một phần backbone với learning rate nhỏ hơn.
- Đánh giá bằng `accuracy`, `loss`, `confusion matrix`.
- Lưu model `.h5`, class map, biểu đồ lịch sử train.

Quy trình dự đoán trong [predict.py](predict.py) và [gui.py](gui.py):
- Đọc ảnh.
- Resize + normalize + BGR->RGB.
- Chạy `model.predict` để lấy vector xác suất 4 lớp.
- Sắp xếp xác suất và hiển thị top-k.

## 6) Cách thức hoạt động của phương pháp

Luồng hoạt động từ đầu đến cuối:

1. Dữ liệu ảnh được tổ chức theo từng lớp trong thư mục train/test.
2. Bộ sinh dữ liệu tự đọc ảnh theo nhãn thư mục và áp dụng augmentation cho train.
3. VGG16 trích xuất đặc trưng thị giác ở nhiều mức độ (cạnh, texture, hình khối).
4. Phần head phân loại học ánh xạ từ đặc trưng sang 4 nhãn phương tiện.
5. Sau khi head ổn định, fine-tune một phần backbone để thích nghi tốt hơn với dữ liệu thực tế.
6. Khi dự đoán ảnh mới, mô hình trả về xác suất từng lớp; lớp có xác suất cao nhất là kết quả top-1.

## 7) Cách chạy nhanh

### Cài thư viện
```bash
pip install -r requirements.txt
```

### Huấn luyện
```bash
python train.py --epochs 15 --batch_size 32
```

### Dự đoán 1 ảnh
```bash
python predict.py duong_dan_anh.png --topk 4
```

### Mở giao diện GUI
```bash
python gui.py
```

## 8) Tệp kết quả sau huấn luyện

Sinh ra trong thư mục `model/`:
- `best_model.h5`
- `traffic_vgg16.h5`
- `class_indices.json`
- `training_history.png`
- `confusion_matrix.png`
- `confusion_matrix.txt`
