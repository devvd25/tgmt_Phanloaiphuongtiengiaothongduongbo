import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify, request, send_from_directory

try:
    from yt_dlp import YoutubeDL
except ImportError:
    YoutubeDL = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (  # noqa: E402
    CLASS_NAMES_VN,
    classify_detected_vehicles,
    detect_vehicle_boxes,
    draw_prediction_text,
    draw_vehicle_detections,
    get_yolo_runtime_info,
    load_class_indices,
    predict_topk_multicrop,
    read_image_bgr,
)

MODEL_PATH = PROJECT_ROOT / "model" / "best_model.h5"
CLASS_MAP_PATH = PROJECT_ROOT / "model" / "class_indices.json"
WEB_DIR = PROJECT_ROOT / "web"
WEB_UPLOADS_DIR = PROJECT_ROOT / "web_uploads"
WEB_OUTPUTS_DIR = PROJECT_ROOT / "web_outputs"
LIVE_LINKS_PATH = WEB_DIR / "saved_live_links.json"

app = Flask(
    __name__,
    static_folder=str(WEB_DIR),
    static_url_path="/web",
)

_MODEL: Optional[tf.keras.Model] = None
_IDX_TO_CLASS: Optional[Dict[int, str]] = None
_MODEL_LOCK = threading.Lock()

TRUCK_MIN_CONF_VIDEO = 0.45
TRUCK_TINY_MIN_CONF_VIDEO = 0.60
TRUCK_TINY_MIN_AREA_RATIO = 0.012
VIDEO_BLACK_MEAN_THRESHOLD = 10.0
VIDEO_BLACK_DARK_RATIO_THRESHOLD = 0.985
VIDEO_TAIL_BLACK_FRAMES_TO_STOP = 12

LIVE_STREAM_MAX_SIDE = 960
LIVE_STREAM_INFER_INTERVAL = 3
LIVE_STREAM_JPEG_QUALITY = 72
LIVE_STREAM_ENABLE_FALLBACK = False
LIVE_SOURCE_TARGET_MAX_HEIGHT = 720

VIDEO_STREAM_SESSION_TTL_SEC = 30 * 60
VIDEO_STREAM_SESSIONS: Dict[str, Dict[str, Any]] = {}
VIDEO_STREAM_SESSIONS_LOCK = threading.Lock()


def ensure_web_dirs() -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    WEB_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    WEB_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup_temp_path(path_value: Optional[str]) -> None:
    if not path_value:
        return

    path = Path(path_value)
    if not path.exists():
        return

    try:
        path.unlink()
    except OSError:
        pass


def cleanup_video_stream_sessions(force: bool = False) -> None:
    now = time.time()
    stale_ids: List[str] = []
    stale_input_paths: List[str] = []

    with VIDEO_STREAM_SESSIONS_LOCK:
        for session_id, session in VIDEO_STREAM_SESSIONS.items():
            updated_at = float(session.get("updated_at", now))
            stream_status = str(session.get("stream_status", "ready")).strip().lower()
            ttl = VIDEO_STREAM_SESSION_TTL_SEC if stream_status in {"ready", "processing"} else 300

            if force or (now - updated_at) > ttl:
                stale_ids.append(session_id)

        for session_id in stale_ids:
            session = VIDEO_STREAM_SESSIONS.pop(session_id, None)
            if not session:
                continue
            stale_input_paths.append(str(session.get("input_path", "")))

    for path_value in stale_input_paths:
        _cleanup_temp_path(path_value)


def _error(message: str, status_code: int = 400):
    return jsonify({"status": "error", "message": message}), status_code


def _ok(payload: Dict[str, Any]):
    return jsonify({"status": "ok", **payload})


def _looks_like_url(value: str) -> bool:
    lower = value.strip().lower()
    return lower.startswith(("http://", "https://", "rtsp://", "rtmp://", "mms://"))


def _safe_int(raw_value: Any, default_value: int, min_value: int, max_value: int) -> int:
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return default_value
    return max(min_value, min(max_value, parsed))


def load_saved_live_links() -> List[str]:
    if not LIVE_LINKS_PATH.exists():
        return []

    try:
        with open(LIVE_LINKS_PATH, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return []

    if not isinstance(raw_data, list):
        return []

    links: List[str] = []
    for item in raw_data:
        if not isinstance(item, str):
            continue
        candidate = item.strip()
        if candidate and _looks_like_url(candidate) and candidate not in links:
            links.append(candidate)

    return links[:50]


def save_live_links(links: List[str]) -> None:
    try:
        with open(LIVE_LINKS_PATH, "w", encoding="utf-8") as f:
            json.dump(links[:50], f, indent=2, ensure_ascii=False)
    except OSError:
        pass


def upsert_live_link(links: List[str], link: str) -> List[str]:
    normalized = link.strip()
    if not normalized:
        return links

    next_links = [item for item in links if item != normalized]
    next_links.insert(0, normalized)
    return next_links[:50]


def delete_live_link(links: List[str], link: str) -> List[str]:
    normalized = link.strip()
    return [item for item in links if item != normalized]


def get_inference_stack() -> Tuple[tf.keras.Model, Dict[int, str]]:
    global _MODEL, _IDX_TO_CLASS

    if _MODEL is not None and _IDX_TO_CLASS is not None:
        return _MODEL, _IDX_TO_CLASS

    with _MODEL_LOCK:
        if _MODEL is not None and _IDX_TO_CLASS is not None:
            return _MODEL, _IDX_TO_CLASS

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy model tại: {MODEL_PATH}")
        if not CLASS_MAP_PATH.exists():
            raise FileNotFoundError(f"Không tìm thấy class map tại: {CLASS_MAP_PATH}")

        model = tf.keras.models.load_model(str(MODEL_PATH))
        class_to_idx = load_class_indices(str(CLASS_MAP_PATH))
        idx_to_class = {int(v): k for k, v in class_to_idx.items()}

        _MODEL = model
        _IDX_TO_CLASS = idx_to_class

    return _MODEL, _IDX_TO_CLASS


def resolve_live_stream_source(live_url: str) -> Tuple[str, str]:
    candidate_url = live_url.strip()
    if not _looks_like_url(candidate_url):
        raise ValueError("Link trực tiếp phải bắt đầu bằng http:// hoặc https://")

    lower_url = candidate_url.lower()
    if lower_url.endswith((".m3u8", ".mpd", ".mp4", ".webm", ".flv")):
        return candidate_url, candidate_url

    if YoutubeDL is None:
        raise ImportError("Thiếu thư viện yt-dlp. Cài bằng lệnh: pip install yt-dlp")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "extract_flat": False,
        "format": (
            f"best[height<={LIVE_SOURCE_TARGET_MAX_HEIGHT}][protocol^=http][vcodec!=none][acodec!=none]/"
            f"best[height<={LIVE_SOURCE_TARGET_MAX_HEIGHT}][vcodec!=none][acodec!=none]/"
            "best[protocol^=http][vcodec!=none][acodec!=none]/best"
        ),
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(candidate_url, download=False)
        if info is None:
            raise RuntimeError("Không lấy được metadata từ link trực tiếp.")

        if isinstance(info, dict) and info.get("entries"):
            entries = [entry for entry in info["entries"] if entry]
            if not entries:
                raise RuntimeError("Link trực tiếp không có entry hợp lệ.")
            info = entries[0]
            if isinstance(info, dict) and info.get("_type") == "url" and info.get("url"):
                info = ydl.extract_info(info["url"], download=False)

    if not isinstance(info, dict):
        raise RuntimeError("Không phân tích được thông tin stream.")

    stream_url = info.get("url")
    if not stream_url:
        formats = info.get("formats") or []
        candidates = [fmt for fmt in formats if isinstance(fmt, dict) and fmt.get("url")]
        if candidates:
            preferred = [
                fmt
                for fmt in candidates
                if (fmt.get("height") or 0) > 0 and (fmt.get("height") or 0) <= LIVE_SOURCE_TARGET_MAX_HEIGHT
            ]
            pick_pool = preferred if preferred else candidates

            best = max(
                pick_pool,
                key=lambda fmt: (
                    fmt.get("height") or 0,
                    fmt.get("tbr") or 0,
                    -(fmt.get("fps") or 0),
                ),
            )
            stream_url = best.get("url")

    if not stream_url:
        raise RuntimeError("Không lấy được URL stream từ link trực tiếp.")

    title = str(info.get("title") or "live_stream").strip()
    return str(stream_url), title


def resize_frame_for_video_inference(frame_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_bgr

    scale = max_side / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def filter_video_detections(detections: List[Dict[str, Any]], image_shape: Tuple[int, int, int]) -> List[Dict[str, Any]]:
    """Suppress noisy truck detections that commonly appear in live/video streams."""
    image_h, image_w = image_shape[:2]
    full_area = float(max(1, image_h * image_w))

    filtered: List[Dict[str, Any]] = []
    for det in detections:
        det_cls = str(det.get("det_class_name", "Unknown"))
        det_conf = float(det.get("det_conf", 0.0))

        if det_cls == "Trucks":
            if det_conf < TRUCK_MIN_CONF_VIDEO:
                continue

            x1, y1, x2, y2 = det.get("box", (0, 0, 0, 0))
            box_area = max(1, max(0, x2 - x1) * max(0, y2 - y1))
            area_ratio = box_area / full_area
            if area_ratio < TRUCK_TINY_MIN_AREA_RATIO and det_conf < TRUCK_TINY_MIN_CONF_VIDEO:
                continue

        filtered.append(det)

    return filtered


def is_frame_mostly_black(
    frame_bgr: np.ndarray,
    mean_threshold: float = VIDEO_BLACK_MEAN_THRESHOLD,
    dark_ratio_threshold: float = VIDEO_BLACK_DARK_RATIO_THRESHOLD,
) -> bool:
    if frame_bgr.size == 0:
        return True

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_value = float(np.mean(gray))
    if mean_value > float(mean_threshold):
        return False

    dark_ratio = float(np.count_nonzero(gray <= 18)) / float(gray.size)
    return dark_ratio >= float(dark_ratio_threshold)


def write_video_with_target_fps(source_path: Path, target_path: Path, target_fps: float) -> bool:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return False

    writer: Optional[cv2.VideoWriter] = None
    written_frames = 0
    success = False

    try:
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            return False

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(target_path), fourcc, max(1.0, float(target_fps)), (w, h))
        if not writer.isOpened():
            writer.release()
            return False

        writer.write(first_frame)
        written_frames = 1

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            written_frames += 1

        success = written_frames > 0
        return success
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if (not success or written_frames <= 0) and target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass


def transcode_video_for_web(source_path: Path, target_path: Path) -> bool:
    """Transcode MP4 to H.264/yuv420p for reliable browser playback."""
    if not source_path.exists() or source_path.stat().st_size <= 0:
        return False
    if imageio_ffmpeg is None:
        return False

    try:
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return False

    cmd = [
        ffmpeg_exe,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(target_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, check=False)
    except OSError:
        return False

    if result.returncode != 0 or (not target_path.exists()) or target_path.stat().st_size <= 0:
        if target_path.exists():
            try:
                target_path.unlink()
            except OSError:
                pass
        return False

    return True


def save_uploaded_file(file_storage, folder: Path, preferred_ext: Optional[str] = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    suffix = preferred_ext or Path(file_storage.filename or "").suffix.lower() or ".bin"
    file_name = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    target_path = folder / file_name
    file_storage.save(str(target_path))
    return target_path


def build_mjpeg_chunk(image_bgr: np.ndarray, quality: int = 82) -> Optional[bytes]:
    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(95, int(quality)))],
    )
    if not ok:
        return None

    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"


def _downscale_for_export(image_bgr: np.ndarray, max_side: Optional[int]) -> np.ndarray:
    if max_side is None or max_side <= 0:
        return image_bgr

    h, w = image_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image_bgr

    scale = max_side / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def save_bgr_image(
    image_bgr: np.ndarray,
    prefix: str,
    ext: str = ".jpg",
    max_side: Optional[int] = None,
) -> str:
    export_bgr = _downscale_for_export(image_bgr, max_side=max_side)

    if ext.lower() == ".png":
        params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
    else:
        ext = ".jpg"
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 92]

    ok, encoded = cv2.imencode(ext, export_bgr, params)
    if not ok:
        raise RuntimeError("Không mã hóa được ảnh kết quả.")

    file_name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    target_path = WEB_OUTPUTS_DIR / file_name
    with open(target_path, "wb") as f:
        f.write(encoded.tobytes())

    return f"/web_outputs/{file_name}"


def resolve_web_output_path_from_url(raw_video_url: str) -> Optional[Path]:
    raw = str(raw_video_url or "").strip()
    if not raw:
        return None

    parsed = urlparse(raw)
    path_value = unquote((parsed.path or raw).strip())
    if not path_value:
        return None

    if path_value.startswith("/web_outputs/"):
        filename = path_value[len("/web_outputs/") :]
    else:
        filename = Path(path_value).name

    safe_name = Path(filename).name
    if not safe_name:
        return None

    return WEB_OUTPUTS_DIR / safe_name


def _normalize_summary_lines(lines: List[str]) -> List[str]:
    return [line for line in lines if line and line.strip()]


def materialize_trace(trace: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    steps = trace.get("steps", [])
    serialized_steps: List[Dict[str, Any]] = []

    for step in steps:
        image_url = None
        image = step.get("image")
        if isinstance(image, np.ndarray):
            trace_max_side = 960 if prefix == "video" else 1280
            image_url = save_bgr_image(
                image,
                prefix=f"trace_{prefix}",
                ext=".jpg",
                max_side=trace_max_side,
            )

        serialized_steps.append(
            {
                "title": step.get("title", "Bước xử lý"),
                "description": step.get("description", ""),
                "image_url": image_url,
            }
        )

    return {
        "title": trace.get("title", "Quy trình xử lý"),
        "summary_lines": _normalize_summary_lines(trace.get("summary_lines", [])),
        "steps": serialized_steps,
    }


def build_trace_for_image_detection(
    original_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    vehicle_preds: List[Dict[str, Any]],
    vis_bgr: np.ndarray,
) -> Dict[str, Any]:
    image_h, image_w = original_bgr.shape[:2]

    steps: List[Dict[str, Any]] = [
        {
            "title": "A. Nhận ảnh đầu vào",
            "description": (
                f"Ảnh gốc được nạp từ file tải lên, kích thước {image_w}x{image_h} px "
                "(định dạng màu BGR của OpenCV)."
            ),
            "image": original_bgr.copy(),
        }
    ]

    det_preview: List[Dict[str, Any]] = []
    for det in detections:
        item = dict(det)
        item["best_class"] = det.get("det_class_name", "Unknown")
        item["best_score"] = float(det.get("det_conf", 0.0))
        det_preview.append(item)

    steps.append(
        {
            "title": "B. Phát hiện phương tiện bằng YOLO",
            "description": (
                f"YOLO quét toàn ảnh và giữ lại {len(detections)} box phương tiện hợp lệ để chuyển sang bước phân loại chi tiết.\n"
                "\n"
                "Giải thích chỉ số trên nhãn (ví dụ: '#1 Xe tải 58.3% | det 43.4%'):\n"
                "1. 58.3%: độ tự tin của nhãn đang hiển thị tại thời điểm này.\n"
                "2. det 43.4%: độ tự tin của detector YOLO rằng box đó thực sự là phương tiện.\n"
                "\n"
                "Lưu ý ở bước B: chưa chạy phân loại crop nên 2 số này thường bằng nhau (đều lấy từ YOLO)."
            ),
            "image": draw_vehicle_detections(original_bgr, det_preview),
        }
    )

    max_steps = 8
    for i, det in enumerate(vehicle_preds[:max_steps], start=1):
        x1, y1, x2, y2 = det["box"]
        crop = original_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        vn_label = CLASS_NAMES_VN.get(det["best_class"], det["best_class"])
        desc = (
            f"Cắt box ({x1},{y1})-({x2},{y2}), resize 224x224 và đưa qua VGG16 để "
            f"chọn lớp có xác suất cao nhất: {vn_label} ({det['best_score'] * 100:.2f}%)."
        )

        steps.append(
            {
                "title": f"C{i}. Phân tích chi tiết xe #{i}",
                "description": desc,
                "image": crop.copy(),
            }
        )

    if len(vehicle_preds) > max_steps:
        steps.append(
            {
                "title": "Y. Rút gọn danh sách minh họa",
                "description": (
                    f"Có thêm {len(vehicle_preds) - max_steps} phương tiện khác, "
                    "được rút gọn trong trace để thuyết trình gọn và dễ theo dõi."
                ),
                "image": None,
            }
        )

    steps.append(
        {
            "title": "Z. Xuất ảnh kết quả cuối",
            "description": "Vẽ box vuông, nhãn tiếng Việt và độ tin cậy trên ảnh để trả ra cho người dùng.",
            "image": vis_bgr.copy(),
        }
    )

    return {
        "title": "Quy trình xử lý ảnh (A-Z, nhiều phương tiện)",
        "summary_lines": [
            "A. Nhận ảnh đầu vào và giải mã bằng OpenCV.",
            "B. YOLO quét toàn ảnh để phát hiện vùng chứa phương tiện.",
            f"C. Số box hợp lệ sau detector: {len(detections)}.",
            "D. Cắt từng box và resize về 224x224 để phù hợp mô hình phân loại VGG16.",
            f"E. Số phương tiện đã phân loại: {len(vehicle_preds)}.",
            "F. Map nhãn từ tiếng Anh sang tiếng Việt để hiển thị thân thiện.",
            "G. Vẽ box + nhãn + confidence và lưu ảnh kết quả cuối.",
            "H. Cách đọc nhãn: '... X% | det Y%' với X là độ tự tin nhãn hiện tại, Y là độ tự tin detector YOLO.",
        ],
        "steps": steps,
    }


def build_trace_for_image_fallback(
    original_bgr: np.ndarray,
    top_preds: List[Any],
    num_views: int,
    vis_bgr: np.ndarray,
) -> Dict[str, Any]:
    best_cls, best_score = top_preds[0]
    best_vn = CLASS_NAMES_VN.get(best_cls, best_cls)
    image_h, image_w = original_bgr.shape[:2]

    top_desc = ", ".join(
        [
            f"{CLASS_NAMES_VN.get(cls_name, cls_name)} {score * 100:.2f}%"
            for cls_name, score in top_preds[:4]
        ]
    )

    return {
        "title": "Quy trình xử lý ảnh (A-Z, fallback toàn ảnh)",
        "summary_lines": [
            "A. Nhận ảnh đầu vào và kiểm tra detector trước.",
            "B. Không có box hợp lệ từ detector, hệ thống kích hoạt fallback.",
            f"C. Multi-crop toàn ảnh với {num_views} vùng nhìn khác nhau.",
            "D. Mỗi vùng crop được resize 224x224 rồi suy luận bằng VGG16.",
            "E. Gộp xác suất từ các vùng để giảm rủi ro lệch do góc chụp.",
            f"F. Kết quả tốt nhất: {best_vn} {best_score * 100:.2f}%.",
            f"G. Top dự đoán tham khảo: {top_desc}.",
        ],
        "steps": [
            {
                "title": "A. Nhận ảnh đầu vào",
                "description": (
                    f"Ảnh kích thước {image_w}x{image_h} px được đưa vào pipeline fallback "
                    "khi detector không trả về box hợp lệ."
                ),
                "image": original_bgr.copy(),
            },
            {
                "title": "B. Quét detector và kích hoạt fallback",
                "description": "Bước detector không tìm thấy phương tiện rõ ràng nên chuyển sang chiến lược quét toàn ảnh.",
                "image": None,
            },
            {
                "title": "C. Multi-crop toàn ảnh",
                "description": (
                    f"Sinh {num_views} crop từ nhiều vùng (giữa, cạnh, góc) để tăng độ bao phủ "
                    "đối tượng khi xe nhỏ hoặc bị che khuất."
                ),
                "image": original_bgr.copy(),
            },
            {
                "title": "D. Tổng hợp và chọn nhãn cuối",
                "description": (
                    f"Gộp xác suất từ tất cả crop, lấy nhãn cao nhất: {best_vn} "
                    f"({best_score * 100:.2f}%)."
                ),
                "image": vis_bgr.copy(),
            },
            {
                "title": "Z. Xuất ảnh fallback",
                "description": "Ảnh kết quả hiển thị nhãn Top-k để người xem nắm được mức độ tự tin của mô hình.",
                "image": vis_bgr.copy(),
            },
        ],
    }


def run_image_pipeline(
    image_bgr: np.ndarray,
    model: tf.keras.Model,
    idx_to_class: Dict[int, str],
    topk: int,
) -> Dict[str, Any]:
    detections = detect_vehicle_boxes(image_bgr=image_bgr)

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

        vis_bgr = draw_vehicle_detections(image_bgr=image_bgr, detections=vehicle_preds)

        lines: List[str] = []
        for i, det in enumerate(vehicle_preds, start=1):
            x1, y1, x2, y2 = det["box"]
            vn_label = CLASS_NAMES_VN.get(det["best_class"], det["best_class"])
            lines.append(
                f"{i}. {vn_label:<10} {det['best_score'] * 100:6.2f}% | box ({x1},{y1})-({x2},{y2})"
            )

        trace = build_trace_for_image_detection(
            original_bgr=image_bgr,
            detections=detections,
            vehicle_preds=vehicle_preds,
            vis_bgr=vis_bgr,
        )

        return {
            "kind": "image",
            "main_text": f"Phát hiện {len(vehicle_preds)} phương tiện trong ảnh.",
            "summary_lines": lines,
            "image_bgr": vis_bgr,
            "trace": trace,
        }

    top_preds, num_views = predict_topk_multicrop(
        model=model,
        image_bgr=image_bgr,
        idx_to_class=idx_to_class,
        image_size=(224, 224),
        top_k=max(1, topk),
    )

    best_class, best_score = top_preds[0]
    vn_name = CLASS_NAMES_VN.get(best_class, best_class)

    lines = []
    for i, (cls_name, score) in enumerate(top_preds, start=1):
        vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
        lines.append(f"Top{i}: {vn_label} {score * 100:.2f}%")

    vis_bgr = draw_prediction_text(image_bgr, lines)

    trace = build_trace_for_image_fallback(
        original_bgr=image_bgr,
        top_preds=top_preds,
        num_views=num_views,
        vis_bgr=vis_bgr,
    )

    return {
        "kind": "image",
        "main_text": f"Fallback toàn ảnh: {vn_name} {best_score * 100:.2f}% (quét {num_views} vùng).",
        "summary_lines": lines,
        "image_bgr": vis_bgr,
        "trace": trace,
    }


def build_trace_for_video(
    *,
    is_live: bool,
    source_name: str,
    frame_count: int,
    frame_has_vehicle: int,
    total_detected_vehicles: int,
    avg_vehicle: float,
    class_counter: Dict[str, int],
    input_frame: Optional[np.ndarray],
    detector_preview: Optional[np.ndarray],
    fallback_preview: Optional[np.ndarray],
    progress_samples: List[Tuple[int, np.ndarray]],
    final_frame: Optional[np.ndarray],
    stopped_early: bool,
    has_replay_result: bool,
) -> Dict[str, Any]:
    mode_text = "trực tiếp" if is_live else "video"
    mode_label = "LIVE realtime" if is_live else "VIDEO file"
    fallback_frame_count = max(0, frame_count - frame_has_vehicle)
    detected_ratio = (frame_has_vehicle / frame_count * 100.0) if frame_count > 0 else 0.0

    summary_lines = [
        f"A. Chế độ xử lý: {mode_label}.",
        f"B. Nguồn đầu vào: {source_name}.",
        f"C. Tổng frame đã xử lý: {frame_count}.",
        f"D. Frame có box phương tiện: {frame_has_vehicle} ({detected_ratio:.1f}%).",
        f"E. Frame dùng fallback toàn ảnh: {fallback_frame_count}.",
        f"F. Tổng số xe phát hiện được: {total_detected_vehicles}.",
        f"G. Trung bình xe/frame có phát hiện: {avg_vehicle:.2f}.",
        "H. Realtime có lớp lọc nhiễu: bỏ box sát biên nhỏ và siết ngưỡng với lớp Truck.",
    ]

    top_classes = sorted(class_counter.items(), key=lambda item: item[1], reverse=True)
    if top_classes:
        top_desc = ", ".join(
            [
                f"{CLASS_NAMES_VN.get(cls_name, cls_name)}: {count}"
                for cls_name, count in top_classes[:4]
            ]
        )
        summary_lines.append(f"I. Top lớp phát hiện: {top_desc}.")
    else:
        summary_lines.append("I. Không có frame nào đủ box chuẩn; hệ thống chạy fallback theo từng frame.")

    summary_lines.append("J. Trạng thái kết thúc: dừng sớm theo yêu cầu/giới hạn frame." if stopped_early else "J. Trạng thái kết thúc: hoàn tất tự nhiên theo luồng dữ liệu.")
    summary_lines.append(
        "K. Đã tạo video replay để tải và xem lại."
        if has_replay_result
        else "K. Không tạo được file replay từ nguồn này."
    )
    summary_lines.append("L. Trace lưu các khung minh họa đầu-cuối và mốc giữa để phục vụ thuyết trình.")

    steps: List[Dict[str, Any]] = []
    if input_frame is not None:
        steps.append(
            {
                "title": "A. Khung đầu vào",
                "description": "Nhận frame từ nguồn video/live, resize để cạnh dài <= 1280 nhằm giữ realtime ổn định.",
                "image": input_frame.copy(),
            }
        )

    if detector_preview is not None:
        steps.append(
            {
                "title": "B. Phát hiện phương tiện bằng YOLO",
                "description": (
                    "YOLO dò box theo từng frame; hệ thống ưu tiên detector cho realtime và lọc bớt nhiễu (đặc biệt với Truck).\n"
                    "\n"
                    "Giải thích chỉ số trên nhãn (ví dụ: '#1 Xe tải 58.3% | det 43.4%'):\n"
                    "1. X% (58.3%): độ tự tin của nhãn đang hiển thị tại frame đó.\n"
                    "2. det Y% (43.4%): độ tự tin detector YOLO rằng box là phương tiện hợp lệ.\n"
                    "\n"
                    "Khi X cao nhưng det thấp: nhãn có thể đúng nhưng box chưa thật sự chắc chắn."
                ),
                "image": detector_preview.copy(),
            }
        )

    if fallback_preview is not None:
        steps.append(
            {
                "title": "C. Fallback theo từng frame",
                "description": "Frame không có box hợp lệ sẽ dùng multi-crop + VGG16 để đưa ra nhãn tham khảo và vẫn giữ luồng realtime liên tục.",
                "image": fallback_preview.copy(),
            }
        )

    for i, (frame_idx, sample_img) in enumerate(progress_samples[:3], start=1):
        steps.append(
            {
                "title": f"D{i}. Mốc theo dõi frame {frame_idx}",
                "description": "Ảnh chụp giữa luồng để kiểm tra độ ổn định nhãn/box và minh họa tiến trình chạy theo thời gian.",
                "image": sample_img.copy(),
            }
        )

    if final_frame is not None:
        steps.append(
            {
                "title": "Z. Khung kết thúc và xuất kết quả",
                "description": "Frame cuối cùng trước khi tổng hợp thống kê, đóng gói replay (nhanh/chậm) và trả payload kết quả.",
                "image": final_frame.copy(),
            }
        )

    if not steps:
        steps.append(
            {
                "title": "A. Khởi tạo luồng suy luận",
                "description": "Hệ thống đã khởi tạo pipeline realtime nhưng chưa lấy được frame minh họa để hiển thị trong trace.",
                "image": None,
            }
        )

    return {
        "title": f"Quy trình xử lý {mode_text} (A-Z)",
        "summary_lines": summary_lines,
        "steps": steps,
    }


def run_video_pipeline(
    source_path: str,
    source_name: str,
    model: tf.keras.Model,
    idx_to_class: Dict[int, str],
    *,
    is_live: bool,
    topk: int,
    max_frames: int,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError("Không mở được nguồn video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    tag = time.strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    mode_tag = "live" if is_live else "video"

    output_fast_path = WEB_OUTPUTS_DIR / f"ket_qua_{mode_tag}_{tag}_{run_id}_nhanh.mp4"
    output_slow_path = WEB_OUTPUTS_DIR / f"ket_qua_{mode_tag}_{tag}_{run_id}_cham.mp4"

    writer: Optional[cv2.VideoWriter] = None
    frame_count = 0
    frame_has_vehicle = 0
    total_detected_vehicles = 0
    class_counter: Dict[str, int] = {}
    last_vis: Optional[np.ndarray] = None
    preview_frame_for_result: Optional[np.ndarray] = None
    stopped_early = False
    trailing_black_frames = 0
    start_time = time.perf_counter()

    trace_input_frame: Optional[np.ndarray] = None
    trace_detector_frame: Optional[np.ndarray] = None
    trace_fallback_frame: Optional[np.ndarray] = None
    trace_progress_samples: List[Tuple[int, np.ndarray]] = []

    try:
        while True:
            if max_frames > 0 and frame_count >= max_frames:
                stopped_early = True
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_for_ai = resize_frame_for_video_inference(frame, max_side=1280)

            # For uploaded video files, skip/stop black-tail frames to avoid ugly all-black endings.
            if (not is_live) and frame_count > 0 and is_frame_mostly_black(frame_for_ai):
                trailing_black_frames += 1
                if trailing_black_frames >= VIDEO_TAIL_BLACK_FRAMES_TO_STOP:
                    stopped_early = True
                    break
                continue

            trailing_black_frames = 0

            if trace_input_frame is None:
                trace_input_frame = frame_for_ai.copy()

            detections = detect_vehicle_boxes(
                image_bgr=frame_for_ai,
                side_ignore_margin_ratio=0.12,
                side_ignore_max_area_ratio=0.04,
                side_ignore_max_conf=0.45,
            )
            detections = filter_video_detections(detections, frame_for_ai.shape)

            if detections:
                display_dets = []
                for det in detections:
                    det_cls = det.get("det_class_name", "Unknown")
                    det_item = dict(det)
                    det_item["best_class"] = det_cls
                    det_item["best_score"] = float(det.get("det_conf", 0.0))
                    display_dets.append(det_item)
                    class_counter[det_cls] = class_counter.get(det_cls, 0) + 1

                vis_frame = draw_vehicle_detections(frame_for_ai, display_dets)
                frame_has_vehicle += 1
                total_detected_vehicles += len(display_dets)
                if trace_detector_frame is None:
                    trace_detector_frame = vis_frame.copy()
                preview_frame_for_result = vis_frame.copy()
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
                    vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
                    lines.append(f"Top{rank}: {vn_label} {score * 100:.1f}%")
                vis_frame = draw_prediction_text(frame_for_ai, lines)
                if trace_fallback_frame is None:
                    trace_fallback_frame = vis_frame.copy()
                if preview_frame_for_result is None and (not is_frame_mostly_black(frame_for_ai)):
                    preview_frame_for_result = vis_frame.copy()

            if writer is None:
                out_h, out_w = vis_frame.shape[:2]
                out_fps = float(fps) if fps and fps > 1 else 24.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_fast_path), fourcc, out_fps, (out_w, out_h))
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    raise RuntimeError("Không tạo được file video kết quả đầu ra.")

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
            writer.write(vis_frame)
            last_vis = vis_frame

            if len(trace_progress_samples) < 3 and (frame_count == 1 or frame_count % 240 == 0):
                trace_progress_samples.append((frame_count, vis_frame.copy()))
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if frame_count <= 0 or last_vis is None:
        if output_fast_path.exists():
            try:
                output_fast_path.unlink()
            except OSError:
                pass
        raise RuntimeError("Nguồn video không có frame hợp lệ để xử lý.")

    final_preview_frame = preview_frame_for_result if preview_frame_for_result is not None else last_vis

    avg_vehicle = (total_detected_vehicles / frame_has_vehicle) if frame_has_vehicle > 0 else 0.0

    remembered_slow: Optional[Path] = None
    has_replay_result = False
    if output_fast_path.exists() and output_fast_path.stat().st_size > 0:
        elapsed_sec = max(1e-6, time.perf_counter() - start_time)
        processing_fps = frame_count / elapsed_sec
        slow_fps = max(2.0, min(12.0, processing_fps))

        slow_ready = write_video_with_target_fps(
            source_path=output_fast_path,
            target_path=output_slow_path,
            target_fps=slow_fps,
        )
        if slow_ready and output_slow_path.exists() and output_slow_path.stat().st_size > 0:
            remembered_slow = output_slow_path

        has_replay_result = True

    trace = build_trace_for_video(
        is_live=is_live,
        source_name=source_name,
        frame_count=frame_count,
        frame_has_vehicle=frame_has_vehicle,
        total_detected_vehicles=total_detected_vehicles,
        avg_vehicle=avg_vehicle,
        class_counter=class_counter,
        input_frame=trace_input_frame,
        detector_preview=trace_detector_frame,
        fallback_preview=trace_fallback_frame,
        progress_samples=trace_progress_samples,
        final_frame=final_preview_frame,
        stopped_early=stopped_early,
        has_replay_result=has_replay_result,
    )

    main_text = (
        f"Dự đoán trực tiếp thành công: {source_name} ({frame_count} frame)."
        if is_live
        else f"Dự đoán video thành công: {source_name} ({frame_count} frame)."
    )

    return {
        "kind": "video",
        "is_live": is_live,
        "main_text": main_text,
        "frame_count": frame_count,
        "frame_has_vehicle": frame_has_vehicle,
        "total_detected_vehicles": total_detected_vehicles,
        "avg_vehicle": avg_vehicle,
        "class_counter": class_counter,
        "preview_bgr": final_preview_frame,
        "fast_video_path": output_fast_path,
        "slow_video_path": remembered_slow,
        "trace": trace,
        "stopped_early": stopped_early,
    }


def serialize_image_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    image_url = save_bgr_image(raw_result["image_bgr"], prefix="ket_qua_anh", ext=".png")
    trace_payload = materialize_trace(raw_result["trace"], prefix="image")

    return {
        "kind": "image",
        "main_text": raw_result["main_text"],
        "summary_lines": _normalize_summary_lines(raw_result.get("summary_lines", [])),
        "image_url": image_url,
        "trace": trace_payload,
    }


def serialize_video_result(raw_result: Dict[str, Any]) -> Dict[str, Any]:
    preview_url = save_bgr_image(raw_result["preview_bgr"], prefix="preview_frame", ext=".jpg")
    trace_payload = materialize_trace(raw_result["trace"], prefix="video")

    fast_video_path = raw_result["fast_video_path"]
    slow_video_path = raw_result.get("slow_video_path")

    fast_url = f"/web_outputs/{fast_video_path.name}" if fast_video_path and fast_video_path.exists() else None
    slow_url = f"/web_outputs/{slow_video_path.name}" if slow_video_path and slow_video_path.exists() else None

    class_counter = raw_result.get("class_counter", {})
    motobikes_count = class_counter.get("Motobikes", 0) + class_counter.get("Motorbikes", 0)
    frame_count = int(raw_result.get("frame_count", 0))
    frame_has_vehicle = int(raw_result.get("frame_has_vehicle", 0))
    fallback_frame_count = max(0, frame_count - frame_has_vehicle)
    detected_ratio = (frame_has_vehicle / frame_count * 100.0) if frame_count > 0 else 0.0

    top_classes = sorted(class_counter.items(), key=lambda item: item[1], reverse=True)
    if top_classes:
        top_desc = ", ".join(
            [
                f"{CLASS_NAMES_VN.get(cls_name, cls_name)}: {count}"
                for cls_name, count in top_classes[:4]
            ]
        )
    else:
        top_desc = "Không có lớp nào đạt điều kiện box trong detector."

    mode_label = "trực tiếp" if bool(raw_result.get("is_live")) else "video"
    ended_label = "Dừng sớm theo yêu cầu/giới hạn." if bool(raw_result.get("stopped_early")) else "Hoàn tất tự nhiên."

    summary_lines = [
        f"A. Chế độ: {mode_label}.",
        f"B. Tổng frame: {frame_count}.",
        f"C. Frame có phát hiện xe: {frame_has_vehicle} ({detected_ratio:.1f}%).",
        f"D. Frame fallback: {fallback_frame_count}.",
        f"E. Tổng số xe detector ghi nhận: {raw_result.get('total_detected_vehicles', 0)}.",
        f"F. Trung bình xe/frame có phát hiện: {raw_result.get('avg_vehicle', 0.0):.2f}.",
        f"G. Xe máy phát hiện: {motobikes_count}.",
        f"H. Top lớp phát hiện: {top_desc}.",
        f"I. Trạng thái kết thúc: {ended_label}",
        "J. Xuất kết quả: bản nhanh + bản chậm (nếu tạo được).",
    ]

    return {
        "kind": "video",
        "is_live": bool(raw_result.get("is_live")),
        "main_text": raw_result.get("main_text", "Dự đoán video thành công."),
        "summary_lines": summary_lines,
        "metrics": {
            "frame_count": int(raw_result.get("frame_count", 0)),
            "frame_has_vehicle": int(raw_result.get("frame_has_vehicle", 0)),
            "total_detected_vehicles": int(raw_result.get("total_detected_vehicles", 0)),
            "motobikes": int(motobikes_count),
        },
        "preview_url": preview_url,
        "fast_video_url": fast_url,
        "slow_video_url": slow_url,
        "trace": trace_payload,
    }


@app.get("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


@app.after_request
def apply_no_cache_headers(response):
    if request.path == "/" or request.path.startswith("/web/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@app.get("/api/health")
def api_health():
    model_ready = MODEL_PATH.exists() and CLASS_MAP_PATH.exists()
    ffmpeg_ready = imageio_ffmpeg is not None
    try:
        tf_gpu_count = len(tf.config.list_physical_devices("GPU"))
    except Exception:
        tf_gpu_count = 0
    yolo_runtime = get_yolo_runtime_info()

    return _ok(
        {
            "message": "Máy chủ web đã sẵn sàng.",
            "model_ready": model_ready,
            "ffmpeg_ready": ffmpeg_ready,
            "tf_gpu_count": tf_gpu_count,
            "yolo_runtime": yolo_runtime,
        }
    )


@app.get("/_stcore/health")
def streamlit_probe_health():
    # Friendly compatibility endpoint to silence stale Streamlit browser probes.
    return jsonify({"status": "ok"})


@app.get("/_stcore/host-config")
def streamlit_probe_host_config():
    # Friendly compatibility endpoint to silence stale Streamlit browser probes.
    return jsonify({})


@app.get("/_stcore/stream")
def streamlit_probe_stream():
    # Friendly compatibility endpoint to silence stale Streamlit browser probes.
    return ("", 204)


@app.get("/api/live-links")
def api_get_live_links():
    links = load_saved_live_links()
    return _ok({"links": links})


@app.post("/api/live-links")
def api_add_live_link():
    payload = request.get_json(silent=True) or {}
    live_url = str(payload.get("url", "")).strip()

    if not live_url:
        return _error("Vui lòng nhập link trực tiếp trước khi lưu.")
    if not _looks_like_url(live_url):
        return _error("Link không hợp lệ. Link phải bắt đầu bằng http:// hoặc https://")

    links = load_saved_live_links()
    links = upsert_live_link(links, live_url)
    save_live_links(links)
    return _ok({"message": "Đã lưu link trực tiếp.", "links": links})


@app.delete("/api/live-links")
def api_delete_live_link():
    live_url = str(request.args.get("url", "")).strip()
    if not live_url:
        return _error("Thiếu tham số url cần xóa.")

    links = load_saved_live_links()
    next_links = delete_live_link(links, live_url)
    save_live_links(next_links)
    return _ok({"message": "Đã xóa link trực tiếp.", "links": next_links})


@app.post("/api/live/resolve")
def api_resolve_live_source():
    payload = request.get_json(silent=True) or {}
    live_url = str(payload.get("live_url", "")).strip()

    if not live_url:
        return _error("Hãy nhập link trực tiếp trước khi chạy.")
    if not _looks_like_url(live_url):
        return _error("Link không hợp lệ. Link phải bắt đầu bằng http:// hoặc https://")

    try:
        stream_source, stream_name = resolve_live_stream_source(live_url)
        return _ok(
            {
                "live_url": live_url,
                "stream_source": stream_source,
                "stream_name": stream_name,
            }
        )
    except Exception as exc:
        return _error(f"Không mở được link trực tiếp: {exc}")


@app.get("/api/live/predict-stream")
def api_live_predict_stream():
    live_url = str(request.args.get("live_url", "")).strip()
    resolved_for_url = str(request.args.get("resolved_for_url", "")).strip()
    pre_resolved_stream_source = str(request.args.get("stream_source", "")).strip()
    pre_resolved_stream_name = str(request.args.get("stream_name", "")).strip()
    topk = _safe_int(request.args.get("topk"), default_value=2, min_value=1, max_value=4)
    max_frames = _safe_int(request.args.get("max_frames"), default_value=0, min_value=0, max_value=20000)

    if not live_url:
        return _error("Hãy nhập link trực tiếp trước khi chạy.")
    if not _looks_like_url(live_url):
        return _error("Link không hợp lệ. Link phải bắt đầu bằng http:// hoặc https://")

    if max_frames == 0:
        max_frames = 0

    try:
        model, idx_to_class = get_inference_stack()
    except Exception as exc:
        return _error(f"Không load được model suy luận: {exc}", status_code=500)

    can_use_pre_resolved = (
        resolved_for_url == live_url
        and bool(pre_resolved_stream_source)
        and _looks_like_url(pre_resolved_stream_source)
    )

    if can_use_pre_resolved:
        stream_source = pre_resolved_stream_source
        stream_name = pre_resolved_stream_name or live_url
    else:
        try:
            stream_source, stream_name = resolve_live_stream_source(live_url)
        except Exception as exc:
            return _error(f"Không mở được link trực tiếp: {exc}")

    links = load_saved_live_links()
    links = upsert_live_link(links, live_url)
    save_live_links(links)

    def generate_mjpeg_stream():
        waiting_frame = np.full((360, 640, 3), 34, dtype=np.uint8)
        cv2.putText(
            waiting_frame,
            "Dang mo luong du doan truc tiep...",
            (26, 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            waiting_frame,
            "Vui long doi frame dau tien",
            (78, 196),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (190, 210, 230),
            1,
            cv2.LINE_AA,
        )
        waiting_chunk = build_mjpeg_chunk(waiting_frame, quality=84)
        if waiting_chunk is not None:
            yield waiting_chunk

        cap = cv2.VideoCapture(stream_source)
        if not cap.isOpened():
            return

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        frame_count = 0
        last_display_dets: List[Dict[str, Any]] = []
        last_fallback_lines: List[str] = []
        try:
            while True:
                if max_frames > 0 and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_for_ai = resize_frame_for_video_inference(frame, max_side=LIVE_STREAM_MAX_SIDE)
                next_frame_idx = frame_count + 1
                should_run_inference = (next_frame_idx <= 2) or (next_frame_idx % LIVE_STREAM_INFER_INTERVAL == 0)

                if should_run_inference:
                    detections = detect_vehicle_boxes(
                        image_bgr=frame_for_ai,
                        side_ignore_margin_ratio=0.12,
                        side_ignore_max_area_ratio=0.04,
                        side_ignore_max_conf=0.45,
                    )
                    detections = filter_video_detections(detections, frame_for_ai.shape)

                    if detections:
                        display_dets = []
                        for det in detections:
                            det_cls = det.get("det_class_name", "Unknown")
                            det_item = dict(det)
                            det_item["best_class"] = det_cls
                            det_item["best_score"] = float(det.get("det_conf", 0.0))
                            display_dets.append(det_item)

                        last_display_dets = display_dets
                        last_fallback_lines = []
                        vis_frame = draw_vehicle_detections(frame_for_ai, display_dets)
                    else:
                        last_display_dets = []

                        if LIVE_STREAM_ENABLE_FALLBACK:
                            top_preds, _ = predict_topk_multicrop(
                                model=model,
                                image_bgr=frame_for_ai,
                                idx_to_class=idx_to_class,
                                image_size=(224, 224),
                                top_k=min(max(1, topk), len(idx_to_class)),
                            )

                            lines = []
                            for rank, (cls_name, score) in enumerate(top_preds, start=1):
                                vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
                                lines.append(f"Top{rank}: {vn_label} {score * 100:.1f}%")
                        else:
                            lines = ["Khong phat hien xe ro o frame nay"]

                        last_fallback_lines = lines
                        vis_frame = draw_prediction_text(frame_for_ai, lines)
                else:
                    if last_display_dets:
                        vis_frame = draw_vehicle_detections(frame_for_ai, last_display_dets)
                    elif last_fallback_lines:
                        vis_frame = draw_prediction_text(frame_for_ai, last_fallback_lines)
                    else:
                        vis_frame = frame_for_ai.copy()

                frame_count += 1
                cv2.putText(
                    vis_frame,
                    f"Live Frame: {frame_count}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                footer = f"Nguon: {stream_name}"[:120]
                cv2.putText(
                    vis_frame,
                    footer,
                    (10, max(32, vis_frame.shape[0] - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )

                if not should_run_inference:
                    cv2.putText(
                        vis_frame,
                        "Realtime Lite",
                        (10, 52),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (220, 220, 220),
                        1,
                        cv2.LINE_AA,
                    )

                chunk = build_mjpeg_chunk(vis_frame, quality=LIVE_STREAM_JPEG_QUALITY)
                if chunk is None:
                    continue

                yield chunk
        except GeneratorExit:
            return
        except Exception:
            return
        finally:
            cap.release()

    response = Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.post("/api/video/prepare-stream")
def api_prepare_video_stream():
    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename is None or uploaded.filename.strip() == "":
        return _error("Hãy chọn file video trước khi dự đoán.")

    topk = _safe_int(request.form.get("topk"), default_value=2, min_value=1, max_value=4)
    max_frames = _safe_int(request.form.get("max_frames"), default_value=0, min_value=0, max_value=20000)

    input_path = save_uploaded_file(uploaded, WEB_UPLOADS_DIR)
    source_name = str(uploaded.filename or "").strip() or input_path.name

    cleanup_video_stream_sessions(force=False)

    session_id = uuid.uuid4().hex
    now = time.time()

    with VIDEO_STREAM_SESSIONS_LOCK:
        VIDEO_STREAM_SESSIONS[session_id] = {
            "session_id": session_id,
            "input_path": str(input_path),
            "source_name": source_name,
            "topk": topk,
            "max_frames": max_frames,
            "stream_status": "ready",
            "stop_requested": False,
            "result": None,
            "error": "",
            "created_at": now,
            "updated_at": now,
        }

    return _ok(
        {
            "session_id": session_id,
            "source_name": source_name,
            "stream_url": f"/api/video/predict-stream?session_id={session_id}",
        }
    )


@app.get("/api/video/stream-result")
def api_video_stream_result():
    session_id = str(request.args.get("session_id", "")).strip()
    if not session_id:
        return _error("Thiếu session_id để lấy kết quả video realtime.")

    with VIDEO_STREAM_SESSIONS_LOCK:
        session = VIDEO_STREAM_SESSIONS.get(session_id)
        if session is None:
            return _error("Không tìm thấy phiên video realtime.", status_code=404)

        payload: Dict[str, Any] = {
            "session_id": session_id,
            "stream_status": str(session.get("stream_status", "ready")),
        }

        if session.get("result") is not None:
            payload["result"] = session["result"]
        if session.get("error"):
            payload["error"] = str(session.get("error"))

    return _ok(payload)


@app.post("/api/video/stop-stream")
def api_video_stop_stream():
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("session_id", "")).strip()
    if not session_id:
        return _error("Thiếu session_id để dừng video realtime.")

    with VIDEO_STREAM_SESSIONS_LOCK:
        session = VIDEO_STREAM_SESSIONS.get(session_id)
        if session is None:
            return _error("Không tìm thấy phiên video realtime.", status_code=404)

        session["stop_requested"] = True
        session["updated_at"] = time.time()
        stream_status = str(session.get("stream_status", "ready"))

    return _ok(
        {
            "session_id": session_id,
            "stream_status": stream_status,
            "message": "Đã gửi yêu cầu dừng video realtime.",
        }
    )


@app.get("/api/video/predict-stream")
def api_video_predict_stream():
    session_id = str(request.args.get("session_id", "")).strip()
    if not session_id:
        return _error("Thiếu session_id để mở luồng dự đoán video.")

    try:
        model, idx_to_class = get_inference_stack()
    except Exception as exc:
        return _error(f"Không load được model suy luận: {exc}", status_code=500)

    with VIDEO_STREAM_SESSIONS_LOCK:
        session = VIDEO_STREAM_SESSIONS.get(session_id)
        if session is None:
            return _error("Không tìm thấy phiên video realtime.", status_code=404)

        stream_status = str(session.get("stream_status", "ready")).strip().lower()
        if stream_status == "processing":
            return _error("Phiên video đang chạy ở kết nối khác.", status_code=409)
        if stream_status in {"done", "stopped", "error"}:
            return _error("Phiên video đã xử lý xong. Hãy bấm Dự đoán lại để tạo phiên mới.", status_code=409)

        source_path = str(session.get("input_path", "")).strip()
        source_name = str(session.get("source_name", "video")).strip() or "video"
        topk = _safe_int(session.get("topk"), default_value=2, min_value=1, max_value=4)
        max_frames = _safe_int(session.get("max_frames"), default_value=0, min_value=0, max_value=20000)

        session["stream_status"] = "processing"
        session["stop_requested"] = False
        session["error"] = ""
        session["result"] = None
        session["updated_at"] = time.time()

    input_path_obj = Path(source_path)

    def _mark_session(**kwargs: Any) -> None:
        with VIDEO_STREAM_SESSIONS_LOCK:
            current = VIDEO_STREAM_SESSIONS.get(session_id)
            if current is None:
                return
            current.update(kwargs)
            current["updated_at"] = time.time()

    def _is_stop_requested() -> bool:
        with VIDEO_STREAM_SESSIONS_LOCK:
            current = VIDEO_STREAM_SESSIONS.get(session_id)
            if current is None:
                return True
            return bool(current.get("stop_requested"))

    def generate_mjpeg_stream():
        waiting_frame = np.full((360, 640, 3), 34, dtype=np.uint8)
        cv2.putText(
            waiting_frame,
            "Dang mo luong du doan video...",
            (36, 154),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            waiting_frame,
            "Vui long doi frame du doan dau tien",
            (56, 196),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (190, 210, 230),
            1,
            cv2.LINE_AA,
        )
        waiting_chunk = build_mjpeg_chunk(waiting_frame, quality=84)
        if waiting_chunk is not None:
            yield waiting_chunk

        cap = cv2.VideoCapture(str(input_path_obj))
        if not cap.isOpened():
            _mark_session(stream_status="error", error="Không mở được video đã tải lên.")
            _cleanup_temp_path(str(input_path_obj))
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        tag = time.strftime("%Y%m%d_%H%M%S")
        run_id = uuid.uuid4().hex[:8]
        output_fast_path = WEB_OUTPUTS_DIR / f"ket_qua_video_{tag}_{run_id}_nhanh.mp4"
        output_slow_path = WEB_OUTPUTS_DIR / f"ket_qua_video_{tag}_{run_id}_cham.mp4"

        writer: Optional[cv2.VideoWriter] = None
        frame_count = 0
        frame_has_vehicle = 0
        total_detected_vehicles = 0
        class_counter: Dict[str, int] = {}
        last_vis: Optional[np.ndarray] = None
        preview_frame_for_result: Optional[np.ndarray] = None
        stopped_early = False
        trailing_black_frames = 0
        start_time = time.perf_counter()

        trace_input_frame: Optional[np.ndarray] = None
        trace_detector_frame: Optional[np.ndarray] = None
        trace_fallback_frame: Optional[np.ndarray] = None
        trace_progress_samples: List[Tuple[int, np.ndarray]] = []

        stream_error: Optional[str] = None

        try:
            while True:
                if _is_stop_requested():
                    stopped_early = True
                    break

                if max_frames > 0 and frame_count >= max_frames:
                    stopped_early = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_for_ai = resize_frame_for_video_inference(frame, max_side=1280)

                # Ignore black-tail frames from uploaded videos to keep final output clean.
                if frame_count > 0 and is_frame_mostly_black(frame_for_ai):
                    trailing_black_frames += 1
                    if trailing_black_frames >= VIDEO_TAIL_BLACK_FRAMES_TO_STOP:
                        stopped_early = True
                        break
                    continue

                trailing_black_frames = 0

                if trace_input_frame is None:
                    trace_input_frame = frame_for_ai.copy()

                detections = detect_vehicle_boxes(
                    image_bgr=frame_for_ai,
                    side_ignore_margin_ratio=0.12,
                    side_ignore_max_area_ratio=0.04,
                    side_ignore_max_conf=0.45,
                )
                detections = filter_video_detections(detections, frame_for_ai.shape)

                if detections:
                    display_dets = []
                    for det in detections:
                        det_cls = det.get("det_class_name", "Unknown")
                        det_item = dict(det)
                        det_item["best_class"] = det_cls
                        det_item["best_score"] = float(det.get("det_conf", 0.0))
                        display_dets.append(det_item)
                        class_counter[det_cls] = class_counter.get(det_cls, 0) + 1

                    vis_frame = draw_vehicle_detections(frame_for_ai, display_dets)
                    frame_has_vehicle += 1
                    total_detected_vehicles += len(display_dets)
                    if trace_detector_frame is None:
                        trace_detector_frame = vis_frame.copy()
                    preview_frame_for_result = vis_frame.copy()
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
                        vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
                        lines.append(f"Top{rank}: {vn_label} {score * 100:.1f}%")

                    vis_frame = draw_prediction_text(frame_for_ai, lines)
                    if trace_fallback_frame is None:
                        trace_fallback_frame = vis_frame.copy()
                    if preview_frame_for_result is None and (not is_frame_mostly_black(frame_for_ai)):
                        preview_frame_for_result = vis_frame.copy()

                if writer is None:
                    out_h, out_w = vis_frame.shape[:2]
                    out_fps = float(fps) if fps and fps > 1 else 24.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(output_fast_path), fourcc, out_fps, (out_w, out_h))
                    if not writer.isOpened():
                        writer.release()
                        writer = None
                        stream_error = "Không tạo được file video kết quả đầu ra."
                        break

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

                footer = f"Nguon: {source_name}"[:120]
                cv2.putText(
                    vis_frame,
                    footer,
                    (10, max(32, vis_frame.shape[0] - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )

                writer.write(vis_frame)
                last_vis = vis_frame

                if len(trace_progress_samples) < 3 and (frame_count == 1 or frame_count % 240 == 0):
                    trace_progress_samples.append((frame_count, vis_frame.copy()))

                chunk = build_mjpeg_chunk(vis_frame, quality=80)
                if chunk is not None:
                    yield chunk

                if frame_count % 30 == 0:
                    _mark_session(stream_status="processing")

        except GeneratorExit:
            stopped_early = True
        except Exception as exc:
            stream_error = f"Lỗi xử lý video realtime: {exc}"
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        _cleanup_temp_path(str(input_path_obj))

        if stream_error is not None:
            if output_fast_path.exists():
                _cleanup_temp_path(str(output_fast_path))
            if output_slow_path.exists():
                _cleanup_temp_path(str(output_slow_path))
            _mark_session(stream_status="error", error=stream_error)
            return

        if frame_count <= 0 or last_vis is None:
            if output_fast_path.exists():
                _cleanup_temp_path(str(output_fast_path))
            if output_slow_path.exists():
                _cleanup_temp_path(str(output_slow_path))

            message = "Đã dừng video trước khi có frame hợp lệ." if stopped_early else "Video không có frame hợp lệ để xử lý."
            _mark_session(stream_status="error", error=message)
            return

        final_preview_frame = preview_frame_for_result if preview_frame_for_result is not None else last_vis

        avg_vehicle = (total_detected_vehicles / frame_has_vehicle) if frame_has_vehicle > 0 else 0.0

        remembered_slow: Optional[Path] = None
        has_replay_result = False
        if output_fast_path.exists() and output_fast_path.stat().st_size > 0:
            elapsed_sec = max(1e-6, time.perf_counter() - start_time)
            processing_fps = frame_count / elapsed_sec
            slow_fps = max(2.0, min(12.0, processing_fps))

            slow_ready = write_video_with_target_fps(
                source_path=output_fast_path,
                target_path=output_slow_path,
                target_fps=slow_fps,
            )
            if slow_ready and output_slow_path.exists() and output_slow_path.stat().st_size > 0:
                remembered_slow = output_slow_path

            has_replay_result = True

        trace = build_trace_for_video(
            is_live=False,
            source_name=source_name,
            frame_count=frame_count,
            frame_has_vehicle=frame_has_vehicle,
            total_detected_vehicles=total_detected_vehicles,
            avg_vehicle=avg_vehicle,
            class_counter=class_counter,
            input_frame=trace_input_frame,
            detector_preview=trace_detector_frame,
            fallback_preview=trace_fallback_frame,
            progress_samples=trace_progress_samples,
            final_frame=final_preview_frame,
            stopped_early=stopped_early,
            has_replay_result=has_replay_result,
        )

        main_text = (
            f"Dự đoán video đã dừng: {source_name} ({frame_count} frame)."
            if stopped_early
            else f"Dự đoán video thành công: {source_name} ({frame_count} frame)."
        )

        raw_result = {
            "kind": "video",
            "is_live": False,
            "main_text": main_text,
            "frame_count": frame_count,
            "frame_has_vehicle": frame_has_vehicle,
            "total_detected_vehicles": total_detected_vehicles,
            "avg_vehicle": avg_vehicle,
            "class_counter": class_counter,
            "preview_bgr": final_preview_frame,
            "fast_video_path": output_fast_path,
            "slow_video_path": remembered_slow,
            "trace": trace,
            "stopped_early": stopped_early,
        }

        payload_result = serialize_video_result(raw_result)
        payload_result["stopped_early"] = bool(stopped_early)
        if stopped_early:
            lines = _normalize_summary_lines(payload_result.get("summary_lines", []))
            lines.append("Video đã dừng theo yêu cầu người dùng.")
            payload_result["summary_lines"] = lines

        _mark_session(
            stream_status="stopped" if stopped_early else "done",
            result=payload_result,
            error="",
        )

    response = Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.post("/api/predict/image")
def api_predict_image():
    model, idx_to_class = get_inference_stack()

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename is None or uploaded.filename.strip() == "":
        return _error("Hãy chọn file ảnh trước khi dự đoán.")

    topk = _safe_int(request.form.get("topk"), default_value=3, min_value=1, max_value=4)

    image_bytes = uploaded.read()
    if not image_bytes:
        return _error("File ảnh tải lên rỗng hoặc không hợp lệ.")

    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return _error("Không giải mã được ảnh tải lên.")

    try:
        raw_result = run_image_pipeline(
            image_bgr=image_bgr,
            model=model,
            idx_to_class=idx_to_class,
            topk=topk,
        )
        payload = serialize_image_result(raw_result)
        return _ok({"result": payload})
    except ImportError as exc:
        return _error(f"Thiếu thư viện: {exc}. Cài bằng lệnh: pip install ultralytics")
    except Exception as exc:
        return _error(f"Lỗi xử lý ảnh: {exc}", status_code=500)


@app.post("/api/predict/video")
def api_predict_video():
    model, idx_to_class = get_inference_stack()

    uploaded = request.files.get("file")
    if uploaded is None or uploaded.filename is None or uploaded.filename.strip() == "":
        return _error("Hãy chọn file video trước khi dự đoán.")

    topk = _safe_int(request.form.get("topk"), default_value=3, min_value=1, max_value=4)
    max_frames = _safe_int(request.form.get("max_frames"), default_value=0, min_value=0, max_value=20000)

    input_path = save_uploaded_file(uploaded, WEB_UPLOADS_DIR)

    try:
        raw_result = run_video_pipeline(
            source_path=str(input_path),
            source_name=input_path.name,
            model=model,
            idx_to_class=idx_to_class,
            is_live=False,
            topk=topk,
            max_frames=max_frames,
        )
        payload = serialize_video_result(raw_result)
        return _ok({"result": payload})
    except ImportError as exc:
        return _error(f"Thiếu thư viện: {exc}. Cài bằng lệnh: pip install ultralytics")
    except Exception as exc:
        return _error(f"Lỗi xử lý video: {exc}", status_code=500)
    finally:
        if input_path.exists():
            try:
                input_path.unlink()
            except OSError:
                pass


@app.post("/api/predict/live")
def api_predict_live():
    model, idx_to_class = get_inference_stack()

    payload = request.get_json(silent=True) or {}
    live_url = str(payload.get("live_url", "")).strip()
    resolved_for_url = str(payload.get("resolved_for_url", "")).strip()
    pre_resolved_stream_source = str(payload.get("stream_source", "")).strip()
    pre_resolved_stream_name = str(payload.get("stream_name", "")).strip()
    topk = _safe_int(payload.get("topk"), default_value=3, min_value=1, max_value=4)
    max_frames = _safe_int(payload.get("max_frames"), default_value=1200, min_value=0, max_value=20000)

    if not live_url:
        return _error("Hãy nhập link trực tiếp trước khi chạy.")
    if not _looks_like_url(live_url):
        return _error("Link không hợp lệ. Link phải bắt đầu bằng http:// hoặc https://")

    if max_frames == 0:
        max_frames = 1200

    can_use_pre_resolved = (
        resolved_for_url == live_url
        and bool(pre_resolved_stream_source)
        and _looks_like_url(pre_resolved_stream_source)
    )

    if can_use_pre_resolved:
        stream_source = pre_resolved_stream_source
        stream_name = pre_resolved_stream_name or live_url
    else:
        try:
            stream_source, stream_name = resolve_live_stream_source(live_url)
        except Exception as exc:
            return _error(f"Không mở được link trực tiếp: {exc}")

    links = load_saved_live_links()
    links = upsert_live_link(links, live_url)
    save_live_links(links)

    try:
        raw_result = run_video_pipeline(
            source_path=stream_source,
            source_name=stream_name,
            model=model,
            idx_to_class=idx_to_class,
            is_live=True,
            topk=topk,
            max_frames=max_frames,
        )
        payload_result = serialize_video_result(raw_result)
        payload_result["saved_links"] = links
        return _ok({"result": payload_result, "links": links})
    except ImportError as exc:
        return _error(f"Thiếu thư viện: {exc}. Cài bằng lệnh: pip install ultralytics")
    except Exception as exc:
        return _error(f"Lỗi xử lý trực tiếp: {exc}", status_code=500)


@app.post("/api/video/ensure-web")
def api_ensure_web_video():
    payload = request.get_json(silent=True) or {}
    raw_video_url = str(payload.get("video_url", "")).strip()
    if not raw_video_url:
        return _error("Thiếu video_url cần tối ưu.")

    source_path = resolve_web_output_path_from_url(raw_video_url)
    if source_path is None:
        return _error("video_url không hợp lệ.")

    if (not source_path.exists()) or source_path.stat().st_size <= 0:
        return _error("Không tìm thấy file video kết quả.", status_code=404)

    if source_path.suffix.lower() != ".mp4":
        return _error("Chỉ hỗ trợ chuyển mã video mp4.")

    if source_path.name.lower().endswith("_web.mp4"):
        return _ok(
            {
                "video_url": f"/web_outputs/{source_path.name}",
                "transcoded": False,
                "message": "Video đã là định dạng web tương thích.",
            }
        )

    web_path = source_path.with_name(f"{source_path.stem}_web.mp4")
    if (not web_path.exists()) or web_path.stat().st_size <= 0:
        converted = transcode_video_for_web(source_path, web_path)
        if not converted:
            return _ok(
                {
                    "video_url": f"/web_outputs/{source_path.name}",
                    "transcoded": False,
                    "message": "Không chuyển mã được sang H.264, giữ nguyên video gốc.",
                }
            )

    return _ok(
        {
            "video_url": f"/web_outputs/{web_path.name}",
            "transcoded": True,
            "message": "Đã tạo video H.264 tương thích trình duyệt.",
        }
    )


@app.get("/web_outputs/<path:filename>")
def web_outputs_file(filename: str):
    return send_from_directory(str(WEB_OUTPUTS_DIR), filename)


@app.get("/api/download/<path:filename>")
def api_download_file(filename: str):
    return send_from_directory(str(WEB_OUTPUTS_DIR), filename, as_attachment=True)


def main() -> None:
    ensure_web_dirs()
    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = _safe_int(os.environ.get("WEB_PORT", "8501"), 8501, 1024, 65535)
    preload_model = str(os.environ.get("WEB_PRELOAD_MODEL", "1")).strip().lower() not in {
        "0",
        "false",
        "no",
    }

    if preload_model:
        try:
            print("Preloading inference stack...")
            get_inference_stack()
            print("Inference stack is ready.")
        except Exception as exc:
            print(f"Warning: preload inference stack failed: {exc}")

    print("=" * 72)
    print("Traffic Classification Web Server (Flask)")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print("Local URL: http://localhost:%d" % port)
    print("=" * 72)

    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
