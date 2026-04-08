import json
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

CLASS_NAMES: List[str] = ["Buses", "Cars", "Motorbikes", "Trucks"]
CLASS_NAMES_VN: Dict[str, str] = {
    "Buses": "Xe bus",
    "Cars": "Ô tô",
    "Motorbikes": "Xe máy",
    "Motobikes": "Xe máy",
    "Trucks": "Xe tải",
    "car": "Ô tô",
    "motorbike": "Xe máy",
    "bus": "Xe bus",
    "truck": "Xe tải",
}

COCO_VEHICLE_CLASS_IDS: List[int] = [2, 3, 5, 7]
COCO_TO_CLASS_NAME: Dict[int, str] = {
    2: "Cars",
    3: "Motobikes",
    5: "Buses",
    7: "Trucks",
}

CLASS_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Cars": (59, 130, 246),
    "Motobikes": (16, 185, 129),
    "Buses": (234, 179, 8),
    "Trucks": (239, 68, 68),
}


def _resolve_class_names(train_dir: str, test_dir: str) -> List[str]:
    # Hỗ trợ cả "Motorbikes" và "Motobikes" để tương thích dữ liệu cũ.
    """Resolve class folder names with backward-compatible support for Motobikes typo."""
    train_subdirs = {name for name in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, name))}
    test_subdirs = {name for name in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, name))}
    all_subdirs = train_subdirs.union(test_subdirs)

    bike_name = "Motorbikes" if "Motorbikes" in all_subdirs else "Motobikes"
    return ["Buses", "Cars", bike_name, "Trucks"]


def create_data_generators(
    train_dir: str,
    test_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    validation_split: float = 0.2,
):
    """Create train/validation/test generators with normalization and augmentation."""
    class_names = _resolve_class_names(train_dir=train_dir, test_dir=test_dir)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split,
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=image_size,
        batch_size=batch_size,
        classes=class_names,
        class_mode="categorical",
        subset="training",
    )

    val_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=image_size,
        batch_size=batch_size,
        classes=class_names,
        class_mode="categorical",
        subset="validation",
    )

    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=image_size,
        batch_size=batch_size,
        classes=class_names,
        class_mode="categorical",
        shuffle=False,
    )

    return train_generator, val_generator, test_generator


def build_vgg16_transfer_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 4,
    learning_rate: float = 1e-4,
    dropout_rate: float = 0.5,
) -> Model:
    """Build transfer learning model based on VGG16 pretrained on ImageNet."""
    base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_class_indices(class_indices: Dict[str, int], save_path: str) -> None:
    """Save class to index mapping for stable inference labels."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, indent=2, ensure_ascii=False)


def load_class_indices(class_indices_path: str) -> Dict[str, int]:
    with open(class_indices_path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_bgr_image_for_model(image_bgr: np.ndarray, image_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    # Chuẩn hóa đầu vào về kích thước 224x224, RGB và giá trị [0,1].
    """Resize and normalize BGR image for model prediction."""
    resized = cv2.resize(image_bgr, image_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched


def read_image_bgr(image_path: str) -> Optional[np.ndarray]:
    # Đọc ảnh có đường dẫn Unicode trên Windows bằng fallback.
    """Read an image robustly, including Windows Unicode paths."""
    image = cv2.imread(image_path)
    if image is not None:
        return image

    # Fallback for paths containing non-ASCII characters on some OpenCV builds.
    try:
        raw = np.fromfile(image_path, dtype=np.uint8)
        if raw.size == 0:
            return None
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return image
    except OSError:
        return None


def predict_topk(
    model: tf.keras.Model,
    image_bgr: np.ndarray,
    idx_to_class: Dict[int, str],
    image_size: Tuple[int, int] = (224, 224),
    top_k: int = 3,
):
    # Dự đoán top-k trên toàn bộ ảnh (không multi-crop).
    """Return top-k class predictions sorted by confidence."""
    input_tensor = preprocess_bgr_image_for_model(image_bgr, image_size=image_size)
    probs = model.predict(input_tensor, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_preds = [(idx_to_class[i], float(probs[i])) for i in top_indices]
    return top_preds


@lru_cache(maxsize=2)
def _get_yolo_detector(model_name: str = "yolov8n.pt"):
    if YOLO is None:
        raise ImportError(
            "Chưa cài thư viện ultralytics. Hãy chạy: pip install ultralytics"
        )
    return YOLO(model_name)


def _to_square_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_w: int,
    image_h: int,
    pad_ratio: float = 0.12,
) -> Tuple[int, int, int, int]:
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    side = int(max(width, height) * (1.0 + pad_ratio))
    side = max(16, side)

    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    sx1 = center_x - side // 2
    sy1 = center_y - side // 2
    sx2 = sx1 + side
    sy2 = sy1 + side

    if sx1 < 0:
        sx2 -= sx1
        sx1 = 0
    if sy1 < 0:
        sy2 -= sy1
        sy1 = 0
    if sx2 > image_w:
        sx1 -= sx2 - image_w
        sx2 = image_w
    if sy2 > image_h:
        sy1 -= sy2 - image_h
        sy2 = image_h

    sx1 = max(0, sx1)
    sy1 = max(0, sy1)

    clipped_side = min(sx2 - sx1, sy2 - sy1)
    sx2 = sx1 + clipped_side
    sy2 = sy1 + clipped_side

    return int(sx1), int(sy1), int(sx2), int(sy2)


def _is_small_side_vehicle(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    det_conf: float,
    image_w: int,
    image_h: int,
    side_margin_ratio: float,
    max_area_ratio: float,
    max_conf: float,
) -> bool:
    # Bỏ xe nhỏ ở sát mép ảnh để tránh đếm nhầm xe trên làn phụ/lề đường.
    if side_margin_ratio <= 0.0 or max_area_ratio <= 0.0:
        return False

    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    area_ratio = (box_w * box_h) / float(max(1, image_w * image_h))
    center_x_ratio = ((x1 + x2) * 0.5) / float(max(1, image_w))

    near_side_edge = center_x_ratio < side_margin_ratio or center_x_ratio > (1.0 - side_margin_ratio)
    is_small = area_ratio < max_area_ratio
    is_low_conf = det_conf < max_conf
    return near_side_edge and is_small and is_low_conf


def _box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def _deduplicate_overlapping_detections(
    detections: List[Dict[str, Any]],
    iou_threshold: float,
) -> List[Dict[str, Any]]:
    # Gộp các box gần như trùng nhau để tránh đếm 1 xe thành nhiều xe.
    kept: List[Dict[str, Any]] = []
    for det in detections:
        current_box = det["box"]
        duplicated = False
        for selected in kept:
            if _box_iou(current_box, selected["box"]) >= iou_threshold:
                duplicated = True
                break
        if not duplicated:
            kept.append(det)
    return kept


def detect_vehicle_boxes(
    image_bgr: np.ndarray,
    model_name: str = "yolov8n.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    max_det: int = 40,
    min_box_size: int = 20,
    side_ignore_margin_ratio: float = 0.20,
    side_ignore_max_area_ratio: float = 0.08,
    side_ignore_max_conf: float = 0.55,
    dedup_iou_threshold: float = 0.90,
) -> List[Dict[str, Any]]:
    """Detect all vehicles in an image and return square bounding boxes.

    Extra filtering removes tiny low-confidence roadside detections and then
    deduplicates near-identical boxes for the same object.
    """
    detector = _get_yolo_detector(model_name=model_name)
    image_h, image_w = image_bgr.shape[:2]

    results = detector.predict(
        source=image_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        classes=COCO_VEHICLE_CLASS_IDS,
        max_det=max_det,
        verbose=False,
    )

    if not results:
        return []

    boxes = results[0].boxes
    if boxes is None:
        return []

    detections: List[Dict[str, Any]] = []
    for box in boxes:
        cls_id = int(box.cls.item())
        det_conf = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        if _is_small_side_vehicle(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            det_conf=det_conf,
            image_w=image_w,
            image_h=image_h,
            side_margin_ratio=side_ignore_margin_ratio,
            max_area_ratio=side_ignore_max_area_ratio,
            max_conf=side_ignore_max_conf,
        ):
            continue

        sx1, sy1, sx2, sy2 = _to_square_box(x1, y1, x2, y2, image_w, image_h)
        if (sx2 - sx1) < min_box_size or (sy2 - sy1) < min_box_size:
            continue

        detections.append(
            {
                "box": (sx1, sy1, sx2, sy2),
                "det_conf": det_conf,
                "det_cls_id": cls_id,
                "det_class_name": COCO_TO_CLASS_NAME.get(cls_id, "Unknown"),
            }
        )

    detections.sort(key=lambda item: item["det_conf"], reverse=True)
    detections = _deduplicate_overlapping_detections(
        detections=detections,
        iou_threshold=dedup_iou_threshold,
    )
    return detections


def classify_detected_vehicles(
    model: tf.keras.Model,
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    idx_to_class: Dict[int, str],
    image_size: Tuple[int, int] = (224, 224),
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Run classifier on each detected vehicle crop."""
    classified: List[Dict[str, Any]] = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        top_preds = predict_topk(
            model=model,
            image_bgr=crop,
            idx_to_class=idx_to_class,
            image_size=image_size,
            top_k=top_k,
        )
        best_class, best_score = top_preds[0]

        item = dict(det)
        item["top_preds"] = top_preds
        item["best_class"] = best_class
        item["best_score"] = best_score
        classified.append(item)

    classified.sort(key=lambda item: (item["box"][1], item["box"][0]))
    return classified


def _bgr_to_rgb_color(color_bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    b, g, r = color_bgr
    return (r, g, b)


@lru_cache(maxsize=16)
def _get_unicode_font(font_size: int) -> ImageFont.ImageFont:
    font_candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "arial.ttf",
        "segoeui.ttf",
        "tahoma.ttf",
        "DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        try:
            if "/" in font_path and not os.path.exists(font_path):
                continue
            return ImageFont.truetype(font_path, size=font_size)
        except OSError:
            continue

    return ImageFont.load_default()


def _draw_label_with_unicode(
    image_bgr: np.ndarray,
    text: str,
    x: int,
    y: int,
    bg_color_bgr: Tuple[int, int, int],
    text_color_rgb: Tuple[int, int, int] = (0, 0, 0),
    font_size: int = 20,
) -> np.ndarray:
    # Dùng Pillow để render tiếng Việt có dấu vì cv2.putText không hỗ trợ Unicode đầy đủ.
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_unicode_font(font_size=font_size)

    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = max(1, text_bbox[2] - text_bbox[0])
    text_h = max(1, text_bbox[3] - text_bbox[1])

    pad_x = 6
    pad_y = 4

    bg_x1 = max(0, x)
    bg_y1 = max(0, y - text_h - (pad_y * 2))
    bg_x2 = min(pil_img.width - 1, bg_x1 + text_w + (pad_x * 2))
    bg_y2 = min(pil_img.height - 1, y + pad_y)

    draw.rectangle(
        [(bg_x1, bg_y1), (bg_x2, bg_y2)],
        fill=_bgr_to_rgb_color(bg_color_bgr),
    )
    draw.text((bg_x1 + pad_x, bg_y1 + pad_y - 1), text, font=font, fill=text_color_rgb)

    output_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return output_bgr


def _rect_intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    return iw * ih


def _label_overlap_score(
    rect: Tuple[int, int, int, int],
    occupied_rects: List[Tuple[int, int, int, int]],
) -> int:
    return sum(_rect_intersection_area(rect, used) for used in occupied_rects)


def _clamp_label_rect(
    x: int,
    y: int,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    x = max(0, min(image_w - w, x))
    y = max(0, min(image_h - h, y))
    return (x, y, x + w, y + h)


def _choose_label_rect_for_box(
    box: Tuple[int, int, int, int],
    label_w: int,
    label_h: int,
    image_w: int,
    image_h: int,
    occupied_rects: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    gap = 6

    # Ưu tiên các vị trí gần box trước, sau đó mới quét bổ sung.
    candidate_points: List[Tuple[int, int]] = [
        (x1, y1 - label_h - gap),
        (x2 - label_w, y1 - label_h - gap),
        (x1, y2 + gap),
        (x2 - label_w, y2 + gap),
        (x1 + 2, y1 + 2),
    ]

    step = label_h + 4
    for i in range(1, 5):
        candidate_points.extend(
            [
                (x1, y1 - label_h - gap - i * step),
                (x1, y2 + gap + (i - 1) * step),
                (x2 - label_w, y2 + gap + (i - 1) * step),
            ]
        )

    best_rect: Optional[Tuple[int, int, int, int]] = None
    best_score: Optional[int] = None

    for cx, cy in candidate_points:
        rect = _clamp_label_rect(cx, cy, label_w, label_h, image_w, image_h)
        score = _label_overlap_score(rect, occupied_rects)

        if score == 0:
            return rect

        if best_score is None or score < best_score:
            best_score = score
            best_rect = rect

    # Nếu vẫn va chạm, quét theo cột gần box để tìm chỗ trống hơn.
    scan_x = max(0, min(image_w - label_w, x1))
    scan_step = max(8, label_h // 2)
    for sy in range(0, max(1, image_h - label_h + 1), scan_step):
        rect = _clamp_label_rect(scan_x, sy, label_w, label_h, image_w, image_h)
        score = _label_overlap_score(rect, occupied_rects)
        if score == 0:
            return rect
        if best_score is None or score < best_score:
            best_score = score
            best_rect = rect

    if best_rect is not None:
        return best_rect

    return _clamp_label_rect(x1, y1, label_w, label_h, image_w, image_h)


def draw_vehicle_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """Draw square boxes and labels for all detected vehicles."""
    output = image_bgr.copy()
    image_h, image_w = output.shape[:2]

    rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font_size = 20
    font = _get_unicode_font(font_size=font_size)
    occupied_label_rects: List[Tuple[int, int, int, int]] = []

    for idx, det in enumerate(detections, start=1):
        x1, y1, x2, y2 = det["box"]
        best_class = det.get("best_class", det.get("det_class_name", "Unknown"))
        best_score = float(det.get("best_score", det.get("det_conf", 0.0)))
        det_conf = float(det.get("det_conf", 0.0))

        vn_name = CLASS_NAMES_VN.get(best_class, best_class)
        color = CLASS_COLORS_BGR.get(best_class, (255, 255, 0))

        draw.rectangle([(x1, y1), (x2, y2)], outline=_bgr_to_rgb_color(color), width=2)

        label = f"#{idx} {vn_name} {best_score * 100:.1f}% | det {det_conf * 100:.1f}%"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = max(1, text_bbox[2] - text_bbox[0])
        text_h = max(1, text_bbox[3] - text_bbox[1])

        pad_x = 6
        pad_y = 4
        label_w = text_w + (pad_x * 2)
        label_h = text_h + (pad_y * 2)

        label_rect = _choose_label_rect_for_box(
            box=(x1, y1, x2, y2),
            label_w=label_w,
            label_h=label_h,
            image_w=image_w,
            image_h=image_h,
            occupied_rects=occupied_label_rects,
        )

        occupied_label_rects.append(label_rect)

        lx1, ly1, lx2, ly2 = label_rect
        draw.rectangle([(lx1, ly1), (lx2, ly2)], fill=_bgr_to_rgb_color(color))
        draw.text((lx1 + pad_x, ly1 + pad_y - 1), label, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _generate_candidate_crop_boxes(image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
    # Tạo danh sách crop ưu tiên khu vực trung tâm/đường phố.
    """Generate center-prior crop boxes (square + rectangular) for wide-image inference."""
    height, width = image_shape[:2]
    min_side = min(height, width)
    scales = [1.0, 0.9, 0.78, 0.66, 0.54]

    boxes: List[Tuple[int, int, int, int]] = []
    for scale in scales:
        crop_size = max(32, int(min_side * scale))
        half = crop_size // 2

        centers = [(width // 2, height // 2)]
        if scale >= 0.7:
            shift = int(0.15 * crop_size)
            centers.extend(
                [
                    (width // 2 - shift, height // 2),
                    (width // 2 + shift, height // 2),
                ]
            )

        # For wide frames, add horizontal exploration near center band.
        if width / max(height, 1) >= 1.25 and scale >= 0.75:
            centers.extend(
                [
                    (int(width * 0.35), height // 2),
                    (int(width * 0.65), height // 2),
                ]
            )

        # Vehicles usually appear in lower-middle area for road scenes.
        if scale <= 0.78:
            centers.append((width // 2, min(height - half, int(height * 0.62))))

        for cx, cy in centers:
            x1 = max(0, min(width - crop_size, cx - half))
            y1 = max(0, min(height - crop_size, cy - half))
            x2 = x1 + crop_size
            y2 = y1 + crop_size
            boxes.append((x1, y1, x2, y2))

    # Add center-prior rectangular boxes for elongated vehicle perspective.
    rect_specs = [(0.92, 0.70), (0.80, 0.58), (0.68, 0.48)]
    for w_ratio, h_ratio in rect_specs:
        cw = max(32, int(width * w_ratio))
        ch = max(32, int(height * h_ratio))

        for cx, cy in [
            (width // 2, height // 2),
            (width // 2, int(height * 0.60)),
        ]:
            x1 = max(0, min(width - cw, cx - cw // 2))
            y1 = max(0, min(height - ch, cy - ch // 2))
            x2 = x1 + cw
            y2 = y1 + ch
            boxes.append((x1, y1, x2, y2))

    # Remove duplicates while keeping deterministic order.
    unique_boxes = list(dict.fromkeys(boxes))
    return unique_boxes


def predict_topk_multicrop(
    model: tf.keras.Model,
    image_bgr: np.ndarray,
    idx_to_class: Dict[int, str],
    image_size: Tuple[int, int] = (224, 224),
    top_k: int = 3,
):
    # Quét nhiều crop và hợp nhất xác suất để tăng độ ổn định trên ảnh rộng.
    """Predict with full-image + multi-crop views, robust for uncropped wide images."""
    height, width = image_bgr.shape[:2]
    full_area = float(max(1, height * width))
    num_classes = len(idx_to_class)

    view_records = []

    boxes = _generate_candidate_crop_boxes(image_bgr.shape)

    for x1, y1, x2, y2 in boxes:
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crop_tensor = preprocess_bgr_image_for_model(crop, image_size=image_size)
        probs = model.predict(crop_tensor, verbose=0)[0]
        conf = float(np.max(probs))
        cls_idx = int(np.argmax(probs))

        area_ratio = ((x2 - x1) * (y2 - y1)) / full_area
        area_weight = np.sqrt(max(1e-6, area_ratio))

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        dx = (center_x - (width / 2.0)) / max(1.0, width / 2.0)
        dy = (center_y - (height / 2.0)) / max(1.0, height / 2.0)
        center_weight = float(np.exp(-1.8 * (dx * dx + dy * dy)))

        uniform_prob = 1.0 / max(1, num_classes)
        sharpness = max(0.0, (conf - uniform_prob) / max(1e-6, 1.0 - uniform_prob))

        # Trọng số theo diện tích, trung tâm và độ sắc dự đoán.
        view_weight = 0.35 * area_weight + 0.35 * center_weight + 0.30 * sharpness
        view_score = conf * (0.55 + 0.45 * center_weight) * (0.55 + 0.45 * sharpness)

        view_records.append(
            {
                "probs": probs,
                "cls": cls_idx,
                "conf": conf,
                "weight": float(view_weight),
                "score": float(view_score),
            }
        )

    if not view_records:
        # Fallback to one full-frame prediction in abnormal cases.
        full_tensor = preprocess_bgr_image_for_model(image_bgr, image_size=image_size)
        probs = model.predict(full_tensor, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_preds = [(idx_to_class[i], float(probs[i])) for i in top_indices]
        return top_preds, 1

    # Giữ lại các crop tốt nhất để giảm nhiễu từ nền ảnh.
    view_records.sort(key=lambda x: x["score"], reverse=True)
    keep_n = min(7, len(view_records))
    selected = view_records[:keep_n]

    sel_weights = np.array([max(1e-6, r["weight"] * r["conf"]) for r in selected], dtype=np.float32)
    sel_probs = np.vstack([r["probs"] for r in selected])
    weighted_probs = np.average(sel_probs, axis=0, weights=sel_weights)

    # Bỏ phiếu theo độ tự tin cao nhất.
    vote_scores = np.zeros(num_classes, dtype=np.float32)
    for rank, rec in enumerate(selected):
        rank_weight = 1.0 / (1.0 + 0.35 * rank)
        vote_scores[rec["cls"]] += float(rec["conf"]) * rank_weight

    vote_sum = float(np.sum(vote_scores))
    vote_probs = vote_scores / vote_sum if vote_sum > 0 else weighted_probs
    best_view_probs = selected[0]["probs"]
    best_view_conf = selected[0]["conf"]

    if best_view_conf >= 0.80:
        # Nếu crop tốt nhất rất tự tin, ưu tiên crop đó.
        probs = 0.55 * best_view_probs + 0.30 * weighted_probs + 0.15 * vote_probs
    else:
        # Nếu không, ưu tiên trung bình có trọng số.
        probs = 0.20 * best_view_probs + 0.60 * weighted_probs + 0.20 * vote_probs

    probs = probs / max(1e-8, float(np.sum(probs)))

    top_indices = np.argsort(probs)[::-1][:top_k]
    top_preds = [(idx_to_class[i], float(probs[i])) for i in top_indices]
    return top_preds, len(view_records)


def draw_prediction_text(
    image_bgr: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    # Vẽ hộp thông tin dự đoán lên ảnh.
    """Draw labeled text block on image for demo visualization."""
    output = image_bgr.copy()
    x, y = origin
    line_height = 32

    rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_unicode_font(font_size=24)

    text_widths = []
    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_widths.append(max(1, text_bbox[2] - text_bbox[0]))

    block_width = max(300, max(text_widths) + 24)
    block_height = line_height * len(lines) + 12

    draw.rectangle(
        [(x - 5, y - 26), (x - 5 + block_width, y - 26 + block_height)],
        fill=(0, 0, 0),
    )
    draw.rectangle(
        [(x - 5, y - 26), (x - 5 + block_width, y - 26 + block_height)],
        outline=(255, 255, 0),
        width=2,
    )

    for i, line in enumerate(lines):
        draw.text(
            (x, y + (i * line_height) - 2),
            line,
            font=font,
            fill=(255, 255, 0),
        )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def plot_training_history(history: tf.keras.callbacks.History, save_dir: str) -> None:
    """Plot and save training curves for loss and accuracy."""
    os.makedirs(save_dir, exist_ok=True)
    history_dict = history.history

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_dict.get("accuracy", []), label="Train Accuracy")
    plt.plot(history_dict.get("val_accuracy", []), label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_dict.get("loss", []), label="Train Loss")
    plt.plot(history_dict.get("val_loss", []), label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    plt.close()


def evaluate_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_dir: str,
) -> np.ndarray:
    """Compute, display and save confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=20)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    np.savetxt(os.path.join(save_dir, "confusion_matrix.txt"), cm, fmt="%d")
    return cm
