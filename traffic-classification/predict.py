import argparse
import os
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import CLASS_NAMES_VN, draw_prediction_text, load_class_indices, predict_topk, predict_topk_multicrop, read_image_bgr
from utils import classify_detected_vehicles, detect_vehicle_boxes, draw_vehicle_detections


def _resize_frame_for_video_inference(frame_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_bgr

    scale = max_side / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict traffic vehicle type from image or webcam.")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image. Ignore when using --webcam.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video file.")
    parser.add_argument("--model", type=str, default="model/best_model.h5", help="Path to trained .h5 model.")
    parser.add_argument(
        "--class_map",
        type=str,
        default="model/class_indices.json",
        help="Path to class index mapping json.",
    )
    parser.add_argument("--webcam", action="store_true", help="Run realtime prediction from webcam.")
    parser.add_argument("--topk", type=int, default=3, help="Show top-k predictions.")
    return parser.parse_args()


def pretty_label(class_name: str, score: float) -> str:
    vn = CLASS_NAMES_VN.get(class_name, class_name)
    en = class_name.capitalize()
    return f"{vn} ({en}) - {score * 100:.2f}%"


def run_image_inference(model, idx_to_class, image_path: str, topk: int = 3):
    # Kiểm tra đường dẫn và đọc ảnh đầu vào.
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image_bgr = read_image_bgr(image_path)
    if image_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    detections = detect_vehicle_boxes(image_bgr=image_bgr)

    print(f"INPUT: ảnh camera ({image_path})")
    print("OUTPUT: loại phương tiện")

    if detections:
        per_vehicle_topk = min(max(1, topk), len(idx_to_class))
        vehicle_preds = classify_detected_vehicles(
            model=model,
            image_bgr=image_bgr,
            detections=detections,
            idx_to_class=idx_to_class,
            image_size=(224, 224),
            top_k=per_vehicle_topk,
        )

        print(f"Phát hiện {len(vehicle_preds)} phương tiện:")
        for rank, det in enumerate(vehicle_preds, start=1):
            best_class = det["best_class"]
            best_score = float(det["best_score"])
            x1, y1, x2, y2 = det["box"]
            print(
                f"  {rank}. {pretty_label(best_class, best_score)} | box=({x1},{y1},{x2},{y2})"
            )

            if per_vehicle_topk > 1:
                for sub_rank, (cls_name, score) in enumerate(det["top_preds"], start=1):
                    print(f"     Top{sub_rank}: {pretty_label(cls_name, score)}")

        vis_bgr = draw_vehicle_detections(image_bgr=image_bgr, detections=vehicle_preds)
    else:
        # Fallback dự đoán toàn ảnh nếu không tìm thấy box phương tiện.
        top_preds, num_views = predict_topk_multicrop(
            model=model,
            image_bgr=image_bgr,
            idx_to_class=idx_to_class,
            image_size=(224, 224),
            top_k=topk,
        )
        best_class, best_score = top_preds[0]

        print("Không phát hiện được box phương tiện, chuyển sang chế độ toàn ảnh.")
        print(pretty_label(best_class, best_score))
        print(f"Chế độ ảnh rộng: quét {num_views} vùng")

        lines = [f"Top1: {pretty_label(best_class, best_score)}"]
        if topk > 1:
            for rank, (cls_name, score) in enumerate(top_preds[1:], start=2):
                lines.append(f"Top{rank}: {cls_name} {score * 100:.1f}%")

        vis_bgr = draw_prediction_text(image_bgr, lines)

    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 6))
    plt.imshow(vis_rgb)
    plt.title("Traffic Vehicle Classification")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_webcam_inference(model, idx_to_class, topk: int = 3):
    print("INPUT: ảnh camera")
    print("OUTPUT: loại phương tiện")
    print("Nhấn 'q' để thoát webcam demo.")

    # Mở webcam và chạy dự đoán theo từng khung hình.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được webcam.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        top_preds = predict_topk(
            model=model,
            image_bgr=frame,
            idx_to_class=idx_to_class,
            image_size=(224, 224),
            top_k=topk,
        )

        best_class, best_score = top_preds[0]
        lines = [f"Top1: {pretty_label(best_class, best_score)}"]

        for rank, (cls_name, score) in enumerate(top_preds[1:], start=2):
            lines.append(f"Top{rank}: {cls_name} {score * 100:.1f}%")

        vis_frame = draw_prediction_text(frame, lines)
        cv2.imshow("Traffic Classification Realtime", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_video_inference(model, idx_to_class, video_path: str, topk: int = 3):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

    print(f"INPUT: video ({video_path})")
    print("OUTPUT: loại phương tiện theo từng frame")
    print("Nhấn 'q' để thoát sớm.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được file video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    wait_ms = int(1000 / fps) if fps and fps > 1 else 30

    frame_count = 0
    frame_has_vehicle = 0
    total_detected_vehicles = 0
    class_counter = Counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_for_ai = _resize_frame_for_video_inference(frame, max_side=1280)

        detections = detect_vehicle_boxes(
            image_bgr=frame_for_ai,
            side_ignore_margin_ratio=0.12,
            side_ignore_max_area_ratio=0.04,
            side_ignore_max_conf=0.45,
        )

        if detections:
            display_dets = []
            for det in detections:
                det_cls = det.get("det_class_name", "Unknown")
                det_item = dict(det)
                det_item["best_class"] = det_cls
                det_item["best_score"] = float(det.get("det_conf", 0.0))
                display_dets.append(det_item)
                class_counter[det_cls] += 1

            vis_frame = draw_vehicle_detections(image_bgr=frame_for_ai, detections=display_dets)
            frame_has_vehicle += 1
            total_detected_vehicles += len(display_dets)
        else:
            top_preds, _ = predict_topk_multicrop(
                model=model,
                image_bgr=frame_for_ai,
                idx_to_class=idx_to_class,
                image_size=(224, 224),
                top_k=min(max(1, topk), len(idx_to_class)),
            )
            lines = []
            for rank, (cls_name, score) in enumerate(top_preds, start=1):
                lines.append(f"Top{rank}: {pretty_label(cls_name, score)}")
            vis_frame = draw_prediction_text(frame_for_ai, lines)

        frame_count += 1
        cv2.putText(
            vis_frame,
            f"Frame: {frame_count}",
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Traffic Classification Video", vis_frame)
        key = cv2.waitKey(max(1, wait_ms)) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if frame_count == 0:
        print("Video không có frame hợp lệ.")
        return

    avg_vehicle = (total_detected_vehicles / frame_has_vehicle) if frame_has_vehicle > 0 else 0.0
    print("=== TÓM TẮT VIDEO ===")
    print(f"Tổng frame: {frame_count}")
    print(f"Frame có phát hiện xe: {frame_has_vehicle}")
    print(f"Tổng số xe đã phát hiện: {total_detected_vehicles}")
    print(f"Trung bình xe/frame có phát hiện: {avg_vehicle:.2f}")
    print(f"Xe máy phát hiện: {class_counter.get('Motobikes', 0)}")


def main():
    args = parse_args()

    if args.webcam and args.video:
        raise ValueError("Không thể dùng đồng thời --webcam và --video")

    # Nạp model và bản đồ lớp đã lưu.
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Không tìm thấy model: {args.model}")
    if not os.path.exists(args.class_map):
        raise FileNotFoundError(f"Không tìm thấy class map: {args.class_map}")

    model = tf.keras.models.load_model(args.model)
    class_to_idx = load_class_indices(args.class_map)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    if args.webcam:
        run_webcam_inference(model, idx_to_class, topk=args.topk)
    elif args.video:
        run_video_inference(model, idx_to_class, video_path=args.video, topk=args.topk)
    else:
        if args.image is None:
            raise ValueError("Vui lòng cung cấp đường dẫn ảnh, hoặc dùng --video/--webcam")
        run_image_inference(model, idx_to_class, args.image, topk=args.topk)


if __name__ == "__main__":
    main()
