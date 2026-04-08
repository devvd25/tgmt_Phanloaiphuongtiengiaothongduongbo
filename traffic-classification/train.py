import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from utils import (
    build_vgg16_transfer_model,
    create_data_generators,
    evaluate_and_save_confusion_matrix,
    plot_training_history,
    save_class_indices,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train traffic vehicle classifier with VGG16 transfer learning.")
    parser.add_argument("--train_dir", type=str, default="dataset/train", help="Path to training dataset root.")
    parser.add_argument("--test_dir", type=str, default="dataset/test", help="Path to test dataset root.")
    parser.add_argument("--model_dir", type=str, default="model", help="Directory to save trained model outputs.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs (recommended 10-20).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Learning rate for fine-tuning stage.")
    parser.add_argument(
        "--freeze_until",
        type=str,
        default="block5_conv1",
        help="VGG16 layer name from which layers become trainable in fine-tuning.",
    )
    parser.add_argument(
        "--disable_fine_tune",
        action="store_true",
        help="Disable second-stage fine-tuning.",
    )
    return parser.parse_args()


def compute_class_weights_from_generator(train_gen):
    # Tính class weights để giảm lệch lớp khi huấn luyện.
    """Compute balanced class weights from training subset to reduce class bias."""
    y = train_gen.classes
    class_counts = np.bincount(y, minlength=train_gen.num_classes)
    total = np.sum(class_counts)

    class_weight = {
        i: float(total / (train_gen.num_classes * class_counts[i])) for i in range(train_gen.num_classes) if class_counts[i] > 0
    }
    return class_weight, class_counts


def merge_histories(history_a, history_b):
    merged = {}
    keys = set(history_a.history.keys()).union(set(history_b.history.keys()))
    for key in keys:
        merged[key] = history_a.history.get(key, []) + history_b.history.get(key, [])

    class CombinedHistory:
        def __init__(self, history_dict):
            self.history = history_dict

    return CombinedHistory(merged)


def unfreeze_vgg16_from_layer(model: tf.keras.Model, freeze_until: str) -> None:
    # Mở khóa dần các layer VGG16 để fine-tune.
    """Unfreeze VGG16 layers from a target layer name for fine-tuning."""
    layer_names = {layer.name for layer in model.layers}
    if freeze_until not in layer_names:
        freeze_until = "block5_conv1"

    trainable_flag = False
    for layer in model.layers:
        # VGG16 backbone layers are named block*_*, keep classifier head trainable.
        if layer.name.startswith("block") or layer.name == "input_layer":
            if layer.name == freeze_until:
                trainable_flag = True
            layer.trainable = trainable_flag


def main():
    args = parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # Tạo bộ sinh dữ liệu train/val/test với augmentation.
    train_gen, val_gen, test_gen = create_data_generators(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        image_size=(224, 224),
        batch_size=args.batch_size,
        validation_split=0.2,
    )

    # Xây dựng model VGG16 transfer learning.
    model = build_vgg16_transfer_model(
        input_shape=(224, 224, 3),
        num_classes=train_gen.num_classes,
        learning_rate=args.lr,
        dropout_rate=0.5,
    )

    class_weight, class_counts = compute_class_weights_from_generator(train_gen)
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    print("Class distribution (training subset):")
    for i in range(train_gen.num_classes):
        print(f"- {idx_to_class[i]}: {int(class_counts[i])} samples, class_weight={class_weight.get(i, 1.0):.4f}")

    # Lưu class map để giữ ổn định nhãn khi dự đoán.
    class_indices_path = os.path.join(args.model_dir, "class_indices.json")
    save_class_indices(train_gen.class_indices, class_indices_path)

    best_model_path = os.path.join(args.model_dir, "best_model.h5")
    final_model_path = os.path.join(args.model_dir, "traffic_vgg16.h5")

    # Callback lưu model tốt nhất và dừng sớm.
    callbacks = [
        ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print("=== STAGE 1: TRAIN CLASSIFIER HEAD ===")
    # Train phần head khi backbone còn đóng băng.
    head_epochs = args.epochs if args.disable_fine_tune else max(1, int(args.epochs * 0.6))
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=head_epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    history = history_head

    if not args.disable_fine_tune and args.epochs > head_epochs:
        print("\n=== STAGE 2: FINE-TUNE VGG16 BACKBONE ===")
        # Fine-tune một phần backbone với learning rate nhỏ hơn.
        unfreeze_vgg16_from_layer(model, freeze_until=args.freeze_until)
        model.compile(
            optimizer=Adam(learning_rate=args.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history_ft = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=args.epochs,
            initial_epoch=head_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        history = merge_histories(history_head, history_ft)

    model.save(final_model_path)

    # Đánh giá trên tập test và lưu confusion matrix.
    print("\n=== EVALUATION ON TEST SET ===")
    test_loss, test_acc = model.evaluate(test_gen, verbose=1)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    labels = [idx_to_class[i] for i in range(len(idx_to_class))]

    cm = evaluate_and_save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        save_dir=args.model_dir,
    )

    plot_training_history(history, save_dir=args.model_dir)

    print("\n=== RESULTS ===")
    print("INPUT: ảnh camera / dataset ảnh")
    print("OUTPUT: loại phương tiện")
    print(f"Saved best model: {best_model_path}")
    print(f"Saved final model: {final_model_path}")
    print(f"Saved class map: {class_indices_path}")
    print(f"Confusion matrix:\n{cm}")


if __name__ == "__main__":
    main()
