import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from utils import CLASS_NAMES_VN, draw_prediction_text, load_class_indices, predict_topk, read_image_bgr


def parse_args():
    parser = argparse.ArgumentParser(description="Predict traffic vehicle type from image or webcam.")
    parser.add_argument("image", nargs="?", default=None, help="Path to input image. Ignore when using --webcam.")
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")

    image_bgr = read_image_bgr(image_path)
    if image_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")

    top_preds = predict_topk(
        model=model,
        image_bgr=image_bgr,
        idx_to_class=idx_to_class,
        image_size=(224, 224),
        top_k=topk,
    )

    best_class, best_score = top_preds[0]

    print(f"INPUT: ảnh camera ({image_path})")
    print("OUTPUT: loại phương tiện")
    print(pretty_label(best_class, best_score))

    if topk > 1:
        print("Top predictions:")
        for rank, (cls_name, score) in enumerate(top_preds, start=1):
            print(f"  {rank}. {pretty_label(cls_name, score)}")

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


def main():
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Không tìm thấy model: {args.model}")
    if not os.path.exists(args.class_map):
        raise FileNotFoundError(f"Không tìm thấy class map: {args.class_map}")

    model = tf.keras.models.load_model(args.model)
    class_to_idx = load_class_indices(args.class_map)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    if args.webcam:
        run_webcam_inference(model, idx_to_class, topk=args.topk)
    else:
        if args.image is None:
            raise ValueError("Vui lòng cung cấp đường dẫn ảnh hoặc dùng --webcam")
        run_image_inference(model, idx_to_class, args.image, topk=args.topk)


if __name__ == "__main__":
    main()
