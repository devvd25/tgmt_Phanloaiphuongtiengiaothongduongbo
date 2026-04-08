import argparse
import hashlib
from pathlib import Path
from typing import Dict, Set, Tuple

from PIL import Image

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CLASS_NAMES = ["Buses", "Cars", "Motobikes", "Trucks"]
PREFIX_MAP = {
    "Buses": "Bus",
    "Cars": "Car",
    "Motobikes": "Motobike",
    "Trucks": "Truck",
}


def default_paths() -> Tuple[Path, Path]:
    script_dir = Path(__file__).resolve().parent
    source_root = script_dir.parent / "archive" / "Dataset"
    target_root = script_dir / "dataset"
    return source_root, target_root


def parse_args() -> argparse.Namespace:
    source_default, target_default = default_paths()
    parser = argparse.ArgumentParser(
        description="Ingest archive dataset into the current dataset with standardized naming."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(source_default),
        help="Source dataset root (contains train/test).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=str(target_default),
        help="Target dataset root (contains train/test).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="*",
        default=["train", "test"],
        help="Splits to ingest (default: train test).",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXT


def max_index(target_dir: Path, prefix: str) -> int:
    max_idx = 0
    if not target_dir.exists():
        return max_idx

    prefix_token = f"{prefix}_"
    for file_path in target_dir.iterdir():
        if not file_path.is_file() or file_path.suffix.lower() != ".png":
            continue
        stem = file_path.stem
        if not stem.startswith(prefix_token):
            continue
        idx_str = stem[len(prefix_token) :]
        if idx_str.isdigit():
            max_idx = max(max_idx, int(idx_str))

    return max_idx


def save_as_png(src_path: Path, dst_path: Path) -> None:
    with Image.open(src_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dst_path, format="PNG")


def image_fingerprint(image: Image.Image) -> str:
    """Create a deterministic hash from pixel content and image size."""
    hasher = hashlib.md5()
    hasher.update(image.width.to_bytes(4, "big"))
    hasher.update(image.height.to_bytes(4, "big"))
    hasher.update(image.tobytes())
    return hasher.hexdigest()


def collect_existing_fingerprints(dst_class_dir: Path) -> Set[str]:
    fingerprints: Set[str] = set()
    if not dst_class_dir.exists():
        return fingerprints

    for file_path in dst_class_dir.iterdir():
        if not is_image_file(file_path):
            continue
        try:
            with Image.open(file_path) as img:
                rgb = img.convert("RGB")
                fingerprints.add(image_fingerprint(rgb))
        except (OSError, ValueError):
            continue

    return fingerprints


def ingest_class(src_class_dir: Path, dst_class_dir: Path, prefix: str) -> Tuple[int, int]:
    dst_class_dir.mkdir(parents=True, exist_ok=True)
    existing_fingerprints = collect_existing_fingerprints(dst_class_dir)
    next_idx = max_index(dst_class_dir, prefix) + 1
    imported = 0
    skipped_duplicate = 0

    for src_path in sorted(src_class_dir.iterdir()):
        if not is_image_file(src_path):
            continue

        dst_path = dst_class_dir / f"{prefix}_{next_idx}.png"
        try:
            with Image.open(src_path) as img:
                rgb = img.convert("RGB")
                fingerprint = image_fingerprint(rgb)

                if fingerprint in existing_fingerprints:
                    skipped_duplicate += 1
                    continue

                rgb.save(dst_path, format="PNG")
        except (OSError, ValueError) as exc:
            print(f"[WARN] Skip {src_path}: {exc}")
            continue

        imported += 1
        existing_fingerprints.add(fingerprint)
        next_idx += 1

    return imported, skipped_duplicate


def ingest_split(source_root: Path, target_root: Path, split: str) -> Dict[str, Tuple[int, int]]:
    src_split_dir = source_root / split
    if not src_split_dir.exists():
        print(f"[WARN] Missing split: {src_split_dir}")
        return {}

    dst_split_dir = target_root / split
    dst_split_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Tuple[int, int]] = {}
    for class_name in CLASS_NAMES:
        src_class_dir = src_split_dir / class_name
        if not src_class_dir.exists():
            continue
        dst_class_dir = dst_split_dir / class_name
        imported, skipped_duplicate = ingest_class(src_class_dir, dst_class_dir, PREFIX_MAP[class_name])
        summary[class_name] = (imported, skipped_duplicate)

    return summary


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    target_root = Path(args.target)

    if not source_root.exists():
        raise FileNotFoundError(f"Source path not found: {source_root}")

    if source_root.resolve() == target_root.resolve():
        raise ValueError("Source and target paths must be different.")

    target_root.mkdir(parents=True, exist_ok=True)

    total_imported = 0
    total_skipped_duplicate = 0
    for split in args.splits:
        summary = ingest_split(source_root, target_root, split)
        for class_name, (imported_count, skipped_duplicate_count) in summary.items():
            print(
                f"[{split}/{class_name}] imported: {imported_count} | duplicate_skipped: {skipped_duplicate_count}"
            )
            total_imported += imported_count
            total_skipped_duplicate += skipped_duplicate_count

    print(f"Done. Total imported files: {total_imported}")
    print(f"Done. Total duplicate skipped: {total_skipped_duplicate}")


if __name__ == "__main__":
    main()
