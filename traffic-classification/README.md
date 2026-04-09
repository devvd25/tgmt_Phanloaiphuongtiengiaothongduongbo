# Traffic Vehicle Classification (VGG16 + YOLO)

README này được cập nhật theo trạng thái dự án hiện tại.

## 1) Tổng quan

Dự án phân loại phương tiện giao thông đường bộ với 4 lớp:

- Buses
- Cars
- Motobikes
- Trucks

Pipeline hiện tại là **YOLO phát hiện nhiều phương tiện** + **VGG16 phân loại từng xe**.

Ứng dụng hỗ trợ:

- Dự đoán ảnh đơn
- Dự đoán video theo frame
- Dự đoán trực tiếp từ link live (YouTube Live, m3u8...)
- GUI tiếng Việt (Tkinter)
- Lưu kết quả ảnh/video
- Xuất video kết quả gồm **2 bản**: nhanh và chậm
- Xem lại video kết quả trong GUI
- Lưu danh sách link live để dùng lại nhanh

## 2) Cấu trúc thư mục chính

```text
traffic-classification/
├── dataset/
│   ├── train/
│   │   ├── Buses/
│   │   ├── Cars/
│   │   ├── Motobikes/
│   │   └── Trucks/
│   └── test/
│       ├── Buses/
│       ├── Cars/
│       ├── Motobikes/
│       └── Trucks/
├── model/
│   ├── best_model.h5
│   ├── traffic_vgg16.h5
│   ├── class_indices.json
│   ├── confusion_matrix.png
│   ├── confusion_matrix.txt
│   └── training_history.png
├── outputs/
├── gui.py
├── ingest_archive_dataset.py
├── normalize_dataset.py
├── predict.py
├── requirements.txt
├── resplit_dataset.py
├── train.py
└── utils.py
```

## 3) Bộ dữ liệu đang dùng

Nguồn dữ liệu được lấy từ Kaggle và thư mục `archive/Dataset`, sau đó nhập/chuẩn hóa về `dataset/`.

Lưu ý tên lớp:

- Code hỗ trợ cả `Motorbikes` và `Motobikes` để tương thích dữ liệu cũ.
- Class map đang lưu trong model hiện tại là:

```json
{
  "Buses": 0,
  "Cars": 1,
  "Motobikes": 2,
  "Trucks": 3
}
```

## 4) Thống kê dữ liệu hiện tại

Số lượng ảnh thực tế trong `dataset/` tại thời điểm cập nhật README:

| Class | Train | Test | Total |
|---|---:|---:|---:|
| Buses | 1956 | 490 | 2446 |
| Cars | 3099 | 775 | 3874 |
| Motobikes | 2836 | 710 | 3546 |
| Trucks | 4020 | 866 | 4886 |
| **Tổng** | **11911** | **2841** | **14752** |

## 5) Tiền xử lý và chuẩn hóa dữ liệu

### Chuẩn hóa thư mục/tên file

Script: `normalize_dataset.py`

Chức năng:

- Đổi tên thư mục lớp về chuẩn: `Buses/Cars/Motorbikes/Trucks`
- Chuyển ảnh về PNG
- Đổi tên ảnh theo prefix (`Bus_*.png`, `Car_*.png`, ...)

Chạy:

```bash
python normalize_dataset.py --dataset dataset
```

### Chia lại train/test có backup và loại trùng

Script: `resplit_dataset.py`

Chức năng:

- Tạo backup trước khi chia lại
- Loại trùng theo MD5
- Chia train/test theo `--test_ratio`

Chạy:

```bash
python resplit_dataset.py --dataset dataset --test_ratio 0.2 --seed 42
```

### Nhập thêm dữ liệu từ archive

Script: `ingest_archive_dataset.py`

Chức năng:

- Nhập ảnh từ `archive/Dataset` vào `dataset`
- Chuẩn hóa sang PNG
- Đặt tên chuẩn theo lớp
- Bỏ qua ảnh trùng bằng fingerprint ảnh

Chạy:

```bash
python ingest_archive_dataset.py --source ../archive/Dataset --target dataset --splits train test
```

## 6) Mô hình và cách hoạt động

### Backbone phân loại

- `VGG16(weights='imagenet', include_top=False)`
- Head: `Flatten -> Dense(256, relu) -> Dropout -> Dense(4, softmax)`

### Huấn luyện

Script: `train.py`

Luồng huấn luyện:

1. Tạo generator train/val/test với augmentation
2. Stage 1: train classifier head (freeze backbone)
3. Stage 2: fine-tune từ layer chỉ định (`block5_conv1` mặc định)
4. Dùng class weights để giảm lệch lớp
5. Lưu model + class map + biểu đồ + confusion matrix

### Suy luận ảnh/video

Script: `predict.py`

Luồng suy luận:

1. YOLO (`ultralytics`) phát hiện xe trong frame/ảnh
2. Mỗi box được chuyển về square box
3. Crop từng xe và đưa qua VGG16 classifier
4. Vẽ box + nhãn + confidence
5. Nếu không phát hiện box: fallback dự đoán toàn ảnh (multi-crop)

## 7) GUI hiện tại

Script: `gui.py`

Tính năng chính:

- Chọn ảnh/video để dự đoán
- Dự đoán trực tiếp từ link (popup nhập link, lưu link, chọn link đã lưu)
- Dừng video bằng nút hoặc phím `q/ESC`
- Kính lúp ảnh
- Click trạng thái để xem trace pipeline xử lý ảnh
- Lưu kết quả:
  - Ảnh: lưu ảnh đã annotate
  - Video: xuất **2 file**
    - `*_nhanh.mp4` (nhanh, giống bản hiện tại)
    - `*_cham.mp4` (chậm, gần tốc độ lúc phân tích)
- Xem lại video kết quả trực tiếp trong GUI (ưu tiên bản chậm)

## 8) Đánh giá mô hình (snapshot hiện tại)

Từ `model/confusion_matrix.txt` (test set 2841 ảnh):

```text
475 7 0 8
4 763 0 8
0 0 710 0
24 5 0 837
```

- Tổng đúng: 2785 / 2841
- Accuracy xấp xỉ: **98.03%**

Lưu ý: số liệu này thay đổi nếu bạn train lại model.

## 9) Cài đặt nhanh

```bash
pip install -r requirements.txt
```

Thư viện chính:

- tensorflow / keras
- opencv-python
- numpy
- matplotlib
- scikit-learn
- Pillow
- ultralytics

## 10) Cách chạy nhanh

### Huấn luyện

```bash
python train.py --epochs 15 --batch_size 32
```

### Dự đoán ảnh

```bash
python predict.py path_to_image.png --topk 4
```

### Dự đoán video

```bash
python predict.py --video path_to_video.mp4 --topk 3
```

### Webcam

```bash
python predict.py --webcam --topk 3
```

### GUI

```bash
python gui.py
```

Trong GUI:

1. Bấm "Dự đoán trực tiếp"
2. Nhập link live bất kỳ hoặc chọn link đã lưu
3. Bấm "Dự đoán trực tiếp" để chạy
