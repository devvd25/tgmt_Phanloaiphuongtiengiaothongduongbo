import argparse
import os
from pathlib import Path

from PIL import Image

FOLDER_RENAME_MAP = {
    "bus": "Buses",
    "car": "Cars",
    "motorbike": "Motorbikes",
    "truck": "Trucks",
}

PREFIX_MAP = {
    "Buses": "Bus",
    "Cars": "Car",
    "Motorbikes": "Motobike",
    "Trucks": "Truck",
}

VALID_READ_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def rename_class_folders(split_dir: Path) -> None:
    for old_name, new_name in FOLDER_RENAME_MAP.items():
        old_path = split_dir / old_name
        new_path = split_dir / new_name
        if old_path.exists() and not new_path.exists():
            old_path.rename(new_path)


def convert_and_rename_images(class_dir: Path, prefix: str) -> int:
    files = sorted([p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_READ_EXT])
    converted = 0

    for idx, file_path in enumerate(files, start=1):
        target_name = f"{prefix}_{idx}.png"
        target_path = class_dir / target_name

        with Image.open(file_path) as img:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(target_path, format="PNG")

        # Keep only standardized PNG output.
        if file_path.resolve() != target_path.resolve():
            file_path.unlink(missing_ok=True)

        converted += 1

    return converted


def normalize_dataset(dataset_root: Path) -> None:
    total_files = 0

    for split in ["train", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue

        rename_class_folders(split_dir)

        for class_name, prefix in PREFIX_MAP.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue

            file_count = convert_and_rename_images(class_dir, prefix)
            total_files += file_count
            print(f"[{split}/{class_name}] standardized files: {file_count}")

    print(f"Done. Total standardized files: {total_files}")


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize dataset folder names and file naming format.")
    parser.add_argument("--dataset", type=str, default="dataset", help="Path to dataset root containing train/test.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_root}")

    normalize_dataset(dataset_root)


if __name__ == "__main__":
    main()
