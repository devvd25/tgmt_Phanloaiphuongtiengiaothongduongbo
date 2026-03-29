import json
import os
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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


def _resolve_class_names(train_dir: str, test_dir: str) -> List[str]:
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
    """Resize and normalize BGR image for model prediction."""
    resized = cv2.resize(image_bgr, image_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched


def read_image_bgr(image_path: str) -> np.ndarray | None:
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
    """Return top-k class predictions sorted by confidence."""
    input_tensor = preprocess_bgr_image_for_model(image_bgr, image_size=image_size)
    probs = model.predict(input_tensor, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_preds = [(idx_to_class[i], float(probs[i])) for i in top_indices]
    return top_preds


def draw_prediction_text(
    image_bgr: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Draw labeled text block on image for demo visualization."""
    output = image_bgr.copy()
    x, y = origin
    line_height = 28

    block_width = max(300, max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for line in lines) + 20)
    block_height = line_height * len(lines) + 10

    cv2.rectangle(output, (x - 5, y - 24), (x - 5 + block_width, y - 24 + block_height), (0, 0, 0), -1)
    cv2.rectangle(output, (x - 5, y - 24), (x - 5 + block_width, y - 24 + block_height), (0, 255, 255), 2)

    for i, line in enumerate(lines):
        cv2.putText(
            output,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return output


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
