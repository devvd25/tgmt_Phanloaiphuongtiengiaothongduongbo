import argparse
import hashlib
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-split dataset into train/test without leakage and with backup."
    )
    parser.add_argument("--dataset", type=str, default="dataset", help="Dataset root containing train and test.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio, e.g. 0.2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    # Tính hash để phát hiện ảnh trùng lặp.
    hasher = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def get_class_names(train_dir: Path, test_dir: Path) -> List[str]:
    # Lấy danh sách lớp từ cả train và test.
    class_names = set()
    if train_dir.exists():
        class_names.update([p.name for p in train_dir.iterdir() if p.is_dir()])
    if test_dir.exists():
        class_names.update([p.name for p in test_dir.iterdir() if p.is_dir()])
    return sorted(class_names)


def collect_unique_files(train_dir: Path, test_dir: Path, class_name: str) -> List[Path]:
    # Gom ảnh từ train/test và loại bỏ ảnh trùng theo MD5.
    candidates: List[Path] = []
    for split_dir in [train_dir / class_name, test_dir / class_name]:
        if not split_dir.exists():
            continue
        for f in split_dir.iterdir():
            if f.is_file() and f.suffix.lower() in VALID_EXT:
                candidates.append(f)

    unique_map: Dict[str, Path] = {}
    for file_path in candidates:
        digest = md5_file(file_path)
        if digest not in unique_map:
            unique_map[digest] = file_path

    return list(unique_map.values())


def split_files(files: List[Path], test_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    # Chia dữ liệu theo tỉ lệ test, bảo đảm còn ít nhất 1 ảnh train.
    random.Random(seed).shuffle(files)
    n_total = len(files)
    n_test = max(1, int(n_total * test_ratio)) if n_total > 1 else 0
    test_files = files[:n_test]
    train_files = files[n_test:]

    if len(train_files) == 0 and len(test_files) > 0:
        train_files = [test_files.pop()]

    return train_files, test_files


def ensure_clean_dir(path: Path) -> None:
    # Xóa thư mục cũ để tạo lại sạch sẽ.
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    # Kiểm tra đủ train/test trước khi resplit.
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Both dataset/train and dataset/test must exist.")

    class_names = get_class_names(train_dir, test_dir)
    if not class_names:
        raise RuntimeError("No class folders found in dataset/train or dataset/test.")

    print("Classes found:", ", ".join(class_names))

    # Sao lưu dữ liệu hiện tại trước khi chia lại.
    backup_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = dataset_root / f"_backup_before_resplit_{backup_tag}"
    backup_train = backup_root / "train"
    backup_test = backup_root / "test"
    backup_root.mkdir(parents=True, exist_ok=True)

    shutil.move(str(train_dir), str(backup_train))
    shutil.move(str(test_dir), str(backup_test))

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for class_name in class_names:
        # Chia lại từng lớp sau khi loại trùng.
        source_files = collect_unique_files(backup_train, backup_test, class_name)
        if not source_files:
            print(f"[WARN] No files found for class {class_name}")
            continue

        train_files, test_files = split_files(source_files, args.test_ratio, args.seed)

        train_class_dir = train_dir / class_name
        test_class_dir = test_dir / class_name
        ensure_clean_dir(train_class_dir)
        ensure_clean_dir(test_class_dir)

        for src in train_files:
            dst = train_class_dir / src.name
            shutil.copy2(src, dst)

        for src in test_files:
            dst = test_class_dir / src.name
            shutil.copy2(src, dst)

        summary.append((class_name, len(source_files), len(train_files), len(test_files)))

    print("\nRe-split complete.")
    print(f"Backup saved at: {backup_root}")
    print("\nClass summary (unique_total, train, test):")
    for class_name, total, n_train, n_test in summary:
        print(f"- {class_name}: total={total}, train={n_train}, test={n_test}")


if __name__ == "__main__":
    main()
