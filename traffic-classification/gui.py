import os
import shutil
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from utils import (
    CLASS_NAMES_VN,
    classify_detected_vehicles,
    detect_vehicle_boxes,
    draw_prediction_text,
    draw_vehicle_detections,
    load_class_indices,
    predict_topk_multicrop,
    read_image_bgr,
)


def _resize_frame_for_video_inference(frame_bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame_bgr

    scale = max_side / float(longest)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _letterbox_center_frame(frame_bgr: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
    if canvas_w <= 0 or canvas_h <= 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return frame_bgr

    scale = min(canvas_w / float(w), canvas_h / float(h))
    if scale <= 0:
        return frame_bgr

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interpolation = cv2.INTER_LINEAR if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((canvas_h, canvas_w, 3), 60, dtype=np.uint8)
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    canvas[off_y : off_y + new_h, off_x : off_x + new_w] = resized
    return canvas


class TrafficClassifierGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Phân loại phương tiện giao thông - VGG16")
        self.root.geometry("980x720")
        self.root.minsize(900, 650)
        self.root.configure(bg="#D9DEE7")

        self.model_path = "model/best_model.h5"
        self.class_map_path = "model/class_indices.json"

        self.model = None
        self.idx_to_class = None

        self.selected_image_path = None
        self.selected_image_bgr = None
        self.selected_video_path: Optional[str] = None
        self.display_image_bgr: Optional[np.ndarray] = None
        self.photo_ref = None
        self.btn_zoom: Optional[ttk.Button] = None
        self.btn_choose_video: Optional[ttk.Button] = None
        self.btn_stop_video: Optional[ttk.Button] = None
        self.btn_save_result: Optional[ttk.Button] = None
        self.btn_replay_result: Optional[ttk.Button] = None
        self.status_label: Optional[ttk.Label] = None
        self.status_click_enabled = False
        self.latest_process_trace: Optional[Dict[str, Any]] = None
        self.trace_window: Optional[tk.Toplevel] = None
        self.trace_photo_refs: List[ImageTk.PhotoImage] = []
        self.zoom_window: Optional[tk.Toplevel] = None
        self.zoom_canvas: Optional[tk.Canvas] = None
        self.zoom_photo_ref: Optional[ImageTk.PhotoImage] = None
        self.zoom_scale = 1.0
        self.zoom_scale_var = tk.StringVar(value="100%")
        self.video_thread: Optional[threading.Thread] = None
        self.video_running = False
        self.video_stop_requested = False
        self.video_window_title = "Traffic Video Prediction - press q or ESC to quit"
        self.replay_window_title = "Replay Analyzed Video - press q or ESC to quit"
        self.last_result_type: Optional[str] = None
        self.last_result_image_bgr: Optional[np.ndarray] = None
        self.last_result_video_path: Optional[str] = None
        self.last_result_video_slow_path: Optional[str] = None

        self._configure_styles()
        self._build_ui()
        self._load_model()
        self.root.protocol("WM_DELETE_WINDOW", self._on_root_close)

    def _configure_styles(self) -> None:
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        self.style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#0E2036", background="#EEF2F8")
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#4A5A73", background="#EEF2F8")
        self.style.configure("CardTitle.TLabel", font=("Segoe UI", 12, "bold"), foreground="#1D2E45", background="#F8FAFD")
        self.style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"), foreground="#0B5ED7", background="#EEF2F8")
        self.style.configure("StatusHint.TLabel", font=("Segoe UI", 11, "bold"), foreground="#0A4BB0", background="#EEF2F8")
        self.style.configure("Result.TLabel", font=("Segoe UI", 16, "bold"), foreground="#0D5C52", background="#F8FAFD")
        self.style.configure("TopTitle.TLabel", font=("Segoe UI", 12, "bold"), foreground="#0E2036", background="#F8FAFD")
        self.style.configure("Note.TLabel", font=("Segoe UI", 10), foreground="#6C757D", background="#F8FAFD")

        self.style.configure(
            "Primary.TButton",
            font=("Segoe UI", 11, "bold"),
            foreground="#FFFFFF",
            background="#1664D4",
            borderwidth=0,
            padding=(16, 10),
        )
        self.style.map("Primary.TButton", background=[("active", "#0F4FB1"), ("disabled", "#8EA9D1")])

        self.style.configure(
            "Ghost.TButton",
            font=("Segoe UI", 11, "bold"),
            foreground="#1D2E45",
            background="#E8EDF5",
            borderwidth=0,
            padding=(16, 10),
        )
        self.style.map("Ghost.TButton", background=[("active", "#D5DEEB"), ("disabled", "#ECEFF5")])

        self.style.configure(
            "Icon.TButton",
            font=("Segoe UI Emoji", 12),
            foreground="#1D2E45",
            background="#E8EDF5",
            borderwidth=0,
            padding=(6, 4),
        )
        self.style.map("Icon.TButton", background=[("active", "#D5DEEB"), ("disabled", "#ECEFF5")])

    def _build_ui(self) -> None:
        container = tk.Frame(self.root, bg="#D9DEE7", padx=20, pady=20)
        container.pack(fill=tk.BOTH, expand=True)

        # 3D-like header card.
        header_shadow = tk.Frame(container, bg="#B6BFCE", bd=0)
        header_shadow.pack(fill=tk.X, pady=(0, 12))
        header_card = tk.Frame(header_shadow, bg="#EEF2F8", relief=tk.RAISED, bd=2)
        header_card.pack(fill=tk.X, padx=(0, 2), pady=(0, 2))

        title = ttk.Label(
            header_card,
            text="Phân loại giao thông (ô tô / xe máy / xe bus / xe tải)",
            style="Title.TLabel",
        )
        title.pack(anchor=tk.W, padx=16, pady=(12, 0))

        subtitle = ttk.Label(
            header_card,
            text="Transfer Learning với VGG16 • 4 lớp • Hiển thị xác suất Top-k",
            style="Subtitle.TLabel",
        )
        subtitle.pack(anchor=tk.W, padx=16, pady=(2, 12))

        controls = tk.Frame(container, bg="#D9DEE7")
        controls.pack(fill=tk.X, pady=(0, 10))

        self.btn_choose = ttk.Button(controls, text="Chọn ảnh", command=self.choose_image, style="Primary.TButton")
        self.btn_choose.pack(side=tk.LEFT)

        self.btn_choose_video = ttk.Button(
            controls,
            text="Chọn video",
            command=self.choose_video,
            style="Ghost.TButton",
        )
        self.btn_choose_video.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_predict = ttk.Button(
            controls,
            text="Dự đoán",
            command=self.predict_image,
            state=tk.DISABLED,
            style="Primary.TButton",
        )
        self.btn_predict.pack(side=tk.LEFT, padx=8)

        self.btn_stop_video = ttk.Button(
            controls,
            text="Dừng video",
            command=self.request_stop_video,
            state=tk.DISABLED,
            style="Ghost.TButton",
        )
        self.btn_stop_video.pack(side=tk.LEFT)

        self.btn_clear = ttk.Button(controls, text="Xóa", command=self.clear_view, style="Ghost.TButton")
        self.btn_clear.pack(side=tk.LEFT)

        self.btn_save_result = ttk.Button(
            controls,
            text="Lưu kết quả",
            command=self.save_result,
            state=tk.DISABLED,
            style="Ghost.TButton",
        )
        self.btn_save_result.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_replay_result = ttk.Button(
            controls,
            text="Xem lại video",
            command=self.replay_result_video,
            state=tk.DISABLED,
            style="Ghost.TButton",
        )
        self.btn_replay_result.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Sẵn sàng. Hãy chọn ảnh để dự đoán.")
        self.status_label = ttk.Label(container, textvariable=self.status_var, style="Status.TLabel", cursor="arrow")
        self.status_label.pack(anchor=tk.W, pady=(4, 10))
        self.status_label.bind("<Button-1>", self._on_status_label_click)

        content = tk.Frame(container, bg="#D9DEE7")
        content.pack(fill=tk.BOTH, expand=True)

        # Left 3D card (input image).
        left_shadow = tk.Frame(content, bg="#B6BFCE")
        left_shadow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left = tk.Frame(left_shadow, bg="#F8FAFD", relief=tk.RAISED, bd=2)
        left.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 2))

        left_title_row = tk.Frame(left, bg="#F8FAFD")
        left_title_row.pack(fill=tk.X, padx=12, pady=(10, 8))

        left_title = ttk.Label(left_title_row, text="Ảnh đầu vào", style="CardTitle.TLabel")
        left_title.pack(side=tk.LEFT)

        self.btn_zoom = ttk.Button(
            left_title_row,
            text="🔍",
            command=self.open_zoom_window,
            state=tk.DISABLED,
            style="Icon.TButton",
            width=3,
        )
        self.btn_zoom.pack(side=tk.RIGHT)

        image_box = tk.Frame(left, bg="#EAF0F8", relief=tk.SUNKEN, bd=1)
        image_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        self.image_label = tk.Label(
            image_box,
            text="Chưa có ảnh",
            anchor=tk.CENTER,
            bg="#EAF0F8",
            fg="#60738F",
            font=("Segoe UI", 12, "bold"),
        )
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Right 3D card (prediction).
        right_shadow = tk.Frame(content, bg="#B6BFCE")
        right_shadow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = tk.Frame(right_shadow, bg="#F8FAFD", relief=tk.RAISED, bd=2)
        right.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 2))

        right_title = ttk.Label(right, text="Kết quả dự đoán", style="CardTitle.TLabel")
        right_title.pack(anchor=tk.W, padx=12, pady=(10, 8))

        self.result_main = ttk.Label(
            right,
            text="OUTPUT: loại phương tiện",
            style="Result.TLabel",
            wraplength=360,
            justify=tk.LEFT,
        )
        self.result_main.pack(anchor=tk.W, padx=12, pady=(4, 12))

        self.top_title = ttk.Label(right, text="Danh sách phương tiện phát hiện:", style="TopTitle.TLabel")
        self.top_title.pack(anchor=tk.W, padx=12)

        self.top_var = tk.StringVar(value="-")
        self.top_label = tk.Label(
            right,
            textvariable=self.top_var,
            justify=tk.LEFT,
            font=("Consolas", 13),
            bg="#F8FAFD",
            fg="#111111",
        )
        self.top_label.pack(anchor=tk.W, padx=12, pady=(6, 0))

        note = ttk.Label(
            right,
            text="INPUT: ảnh camera\nOUTPUT: loại phương tiện",
            style="Note.TLabel",
            justify=tk.LEFT,
        )
        note.pack(side=tk.BOTTOM, anchor=tk.W, padx=12, pady=(20, 12))

    def _set_status(self, text: str, clickable: bool = False) -> None:
        self.status_click_enabled = clickable
        if clickable:
            self.status_var.set(f"{text} (Bấm vào đây để xem quy trình xử lý)")
            if self.status_label is not None:
                self.status_label.configure(style="StatusHint.TLabel", cursor="hand2")
        else:
            self.status_var.set(text)
            if self.status_label is not None:
                self.status_label.configure(style="Status.TLabel", cursor="arrow")

    def _set_save_button_enabled(self, enabled: bool) -> None:
        if self.btn_save_result is None:
            return
        self.btn_save_result.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def _set_replay_button_enabled(self, enabled: bool) -> None:
        if self.btn_replay_result is None:
            return
        self.btn_replay_result.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def _reset_saved_result_state(self, clean_temp_video: bool = True) -> None:
        if clean_temp_video:
            self._cleanup_temp_video_result()
        self.last_result_type = None
        self.last_result_image_bgr = None
        self.last_result_video_path = None
        self.last_result_video_slow_path = None
        self._set_save_button_enabled(False)
        self._set_replay_button_enabled(False)

    def _cleanup_temp_video_result(self) -> None:
        for temp_path in [self.last_result_video_path, self.last_result_video_slow_path]:
            if not temp_path:
                continue

            try:
                basename = os.path.basename(temp_path)
                if basename.startswith("_tmp_result_video_") and os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _remember_image_result(self, image_bgr: np.ndarray) -> None:
        self._cleanup_temp_video_result()
        self.last_result_video_path = None
        self.last_result_video_slow_path = None
        self.last_result_type = "image"
        self.last_result_image_bgr = image_bgr.copy()
        self._set_save_button_enabled(True)
        self._set_replay_button_enabled(False)

    def _remember_video_result(self, video_fast_path: str, video_slow_path: Optional[str] = None) -> None:
        self._cleanup_temp_video_result()
        self.last_result_image_bgr = None
        self.last_result_type = "video"
        self.last_result_video_path = video_fast_path
        self.last_result_video_slow_path = video_slow_path
        self._set_save_button_enabled(True)
        self._set_replay_button_enabled(True)

    def _save_image_to_path(self, image_bgr: np.ndarray, save_path: str) -> str:
        ext = os.path.splitext(save_path)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            ext = ".png"
            save_path = save_path + ext

        ok, encoded = cv2.imencode(ext, image_bgr)
        if not ok:
            raise RuntimeError("Không mã hóa được ảnh để lưu.")
        encoded.tofile(save_path)
        return save_path

    def _write_video_with_target_fps(self, source_path: str, target_path: str, target_fps: float) -> bool:
        cap = cv2.VideoCapture(source_path)
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
            writer = cv2.VideoWriter(target_path, fourcc, max(1.0, float(target_fps)), (w, h))
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
            if (not success or written_frames <= 0) and os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except OSError:
                    pass

    def save_result(self) -> None:
        if self.last_result_type == "image" and self.last_result_image_bgr is not None:
            default_name = f"ket_qua_anh_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            path = filedialog.asksaveasfilename(
                title="Lưu ảnh đã phân tích",
                defaultextension=".png",
                initialfile=default_name,
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("BMP", "*.bmp"),
                    ("WEBP", "*.webp"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return

            try:
                saved_path = self._save_image_to_path(self.last_result_image_bgr, path)
            except Exception as exc:
                messagebox.showerror("Lỗi", f"Không lưu được ảnh kết quả.\nChi tiết: {exc}")
                return

            messagebox.showinfo("Thành công", f"Đã lưu ảnh kết quả:\n{saved_path}")
            return

        if self.last_result_type == "video" and self.last_result_video_path:
            if not os.path.exists(self.last_result_video_path):
                messagebox.showerror("Lỗi", "Không tìm thấy video kết quả tạm để lưu.")
                return

            ext = os.path.splitext(self.last_result_video_path)[1].lower() or ".mp4"
            default_name = f"ket_qua_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
            path = filedialog.asksaveasfilename(
                title="Lưu video đã phân tích (sẽ xuất bản nhanh + chậm)",
                defaultextension=ext,
                initialfile=default_name,
                filetypes=[
                    ("MP4", "*.mp4"),
                    ("AVI", "*.avi"),
                    ("MOV", "*.mov"),
                    ("MKV", "*.mkv"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return

            base_path, user_ext = os.path.splitext(path)
            if user_ext.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
                user_ext = ext
                base_path = path

            fast_save_path = f"{base_path}_nhanh{user_ext}"
            slow_save_path = f"{base_path}_cham{user_ext}"

            try:
                shutil.copy2(self.last_result_video_path, fast_save_path)

                slow_source_path = self.last_result_video_slow_path
                if slow_source_path and os.path.exists(slow_source_path):
                    shutil.copy2(slow_source_path, slow_save_path)
                else:
                    created = self._write_video_with_target_fps(
                        source_path=self.last_result_video_path,
                        target_path=slow_save_path,
                        target_fps=6.0,
                    )
                    if not created:
                        shutil.copy2(self.last_result_video_path, slow_save_path)
            except Exception as exc:
                messagebox.showerror("Lỗi", f"Không lưu được video kết quả (nhanh + chậm).\nChi tiết: {exc}")
                return

            messagebox.showinfo(
                "Thành công",
                "Đã lưu 2 bản video kết quả:\n"
                f"- Nhanh: {fast_save_path}\n"
                f"- Chậm: {slow_save_path}",
            )
            return

        messagebox.showinfo("Thông báo", "Chưa có kết quả phân tích để lưu.")

    def _play_video_file(self, video_path: str, window_title: str) -> None:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được video để phát lại.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = int(1000 / fps) if fps and fps > 1 else 30

        frame_count = 0
        window_initialized = False
        stopped_early = False

        self.root.update_idletasks()
        screen_w = max(1, int(self.root.winfo_screenwidth()))
        screen_h = max(1, int(self.root.winfo_screenheight()))

        self.video_running = True
        self.video_stop_requested = False
        self._set_video_controls_busy(True)
        self._set_status("Đang phát lại video kết quả...", clickable=False)

        try:
            while True:
                if self.video_stop_requested:
                    stopped_early = True
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                if not window_initialized:
                    try:
                        self._setup_video_output_window(frame.shape, screen_w, screen_h, window_title=window_title)
                    except cv2.error:
                        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    window_initialized = True

                display_frame = frame
                if window_initialized:
                    try:
                        _, _, win_w, win_h = cv2.getWindowImageRect(window_title)
                        if win_w > 0 and win_h > 0:
                            display_frame = _letterbox_center_frame(frame, win_w, win_h)
                    except cv2.error:
                        pass

                cv2.putText(
                    display_frame,
                    f"Replay Frame: {frame_count}",
                    (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_title, display_frame)

                key = cv2.waitKey(max(1, wait_ms)) & 0xFF
                if key in (ord("q"), 27):
                    stopped_early = True
                    break
        finally:
            cap.release()
            try:
                cv2.destroyWindow(window_title)
            except cv2.error:
                pass

            self.video_running = False
            self.video_stop_requested = False
            self._set_video_controls_busy(False)

        if frame_count == 0:
            messagebox.showwarning("Cảnh báo", "Video kết quả không có frame để phát lại.")
            return

        suffix = " (dừng sớm)" if stopped_early else ""
        self._set_status(f"Đã xem lại video kết quả ({frame_count} frame){suffix}.", clickable=False)

    def replay_result_video(self) -> None:
        if self.last_result_type != "video" or (
            not self.last_result_video_slow_path and not self.last_result_video_path
        ):
            messagebox.showinfo("Thông báo", "Chưa có video kết quả để xem lại.")
            return

        replay_path = self.last_result_video_slow_path or self.last_result_video_path
        if replay_path is None or (not os.path.exists(replay_path)):
            self._set_replay_button_enabled(False)
            messagebox.showerror("Lỗi", "Không tìm thấy file video kết quả để xem lại.")
            return

        if self.video_running:
            messagebox.showinfo("Thông báo", "Đang có video chạy, hãy dừng trước khi xem lại.")
            return

        self._play_video_file(
            video_path=replay_path,
            window_title=self.replay_window_title,
        )

    def _set_video_controls_busy(self, busy: bool) -> None:
        if self.btn_choose is not None:
            self.btn_choose.config(state=tk.DISABLED if busy else tk.NORMAL)
        if self.btn_choose_video is not None:
            self.btn_choose_video.config(state=tk.DISABLED if busy else tk.NORMAL)
        if self.btn_clear is not None:
            self.btn_clear.config(state=tk.DISABLED if busy else tk.NORMAL)

        if self.btn_stop_video is not None:
            self.btn_stop_video.config(state=tk.NORMAL if busy else tk.DISABLED)

        if self.btn_replay_result is not None:
            if busy:
                self.btn_replay_result.config(state=tk.DISABLED)
            else:
                can_replay = (
                    self.last_result_type == "video"
                    and (
                        (bool(self.last_result_video_slow_path) and os.path.exists(self.last_result_video_slow_path))
                        or (bool(self.last_result_video_path) and os.path.exists(self.last_result_video_path))
                    )
                )
                self.btn_replay_result.config(state=tk.NORMAL if can_replay else tk.DISABLED)

        if busy:
            if self.btn_predict is not None:
                self.btn_predict.config(state=tk.DISABLED)
            if self.btn_zoom is not None:
                self.btn_zoom.config(state=tk.DISABLED)
            return

        has_input = (self.selected_image_bgr is not None) or bool(self.selected_video_path)
        if self.btn_predict is not None:
            self.btn_predict.config(state=tk.NORMAL if has_input else tk.DISABLED)

        if self.btn_zoom is not None:
            self.btn_zoom.config(state=tk.NORMAL if self.display_image_bgr is not None else tk.DISABLED)

    def request_stop_video(self) -> None:
        if not self.video_running:
            return
        self.video_stop_requested = True
        self._set_status("Đang dừng video...", clickable=False)

    def _on_root_close(self) -> None:
        if self.video_running:
            self.video_stop_requested = True
        self._cleanup_temp_video_result()
        self._close_zoom_window()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        self.root.destroy()

    def _to_photo_image(self, image_bgr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((max_w, max_h))
        return ImageTk.PhotoImage(pil_img)

    def _clamp_zoom_scale(self, value: float) -> float:
        return max(0.2, min(6.0, value))

    def _render_zoom_canvas(self) -> None:
        if self.zoom_canvas is None or self.display_image_bgr is None:
            return

        h, w = self.display_image_bgr.shape[:2]
        scaled_w = max(1, int(w * self.zoom_scale))
        scaled_h = max(1, int(h * self.zoom_scale))

        interpolation = cv2.INTER_LINEAR if self.zoom_scale >= 1.0 else cv2.INTER_AREA
        resized = cv2.resize(self.display_image_bgr, (scaled_w, scaled_h), interpolation=interpolation)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.zoom_photo_ref = ImageTk.PhotoImage(pil_img)

        self.zoom_canvas.delete("all")
        self.zoom_canvas.create_image(0, 0, anchor=tk.NW, image=self.zoom_photo_ref)
        self.zoom_canvas.configure(scrollregion=(0, 0, scaled_w, scaled_h))
        self.zoom_scale_var.set(f"{self.zoom_scale * 100:.0f}%")

    def _zoom_in(self) -> None:
        self.zoom_scale = self._clamp_zoom_scale(self.zoom_scale * 1.15)
        self._render_zoom_canvas()

    def _zoom_out(self) -> None:
        self.zoom_scale = self._clamp_zoom_scale(self.zoom_scale / 1.15)
        self._render_zoom_canvas()

    def _zoom_reset(self) -> None:
        self.zoom_scale = 1.0
        self._render_zoom_canvas()

    def _zoom_fit(self) -> None:
        if self.zoom_canvas is None or self.display_image_bgr is None:
            return

        canvas_w = max(1, self.zoom_canvas.winfo_width() - 16)
        canvas_h = max(1, self.zoom_canvas.winfo_height() - 16)
        h, w = self.display_image_bgr.shape[:2]

        fit_scale = min(canvas_w / max(1, w), canvas_h / max(1, h))
        self.zoom_scale = self._clamp_zoom_scale(fit_scale)
        self._render_zoom_canvas()

    def _on_zoom_mousewheel(self, event) -> None:
        delta = 0
        if hasattr(event, "delta") and event.delta:
            delta = 1 if event.delta > 0 else -1
        elif hasattr(event, "num"):
            if event.num == 4:
                delta = 1
            elif event.num == 5:
                delta = -1

        if delta > 0:
            self._zoom_in()
        elif delta < 0:
            self._zoom_out()

    def _close_zoom_window(self) -> None:
        if self.zoom_window is not None and self.zoom_window.winfo_exists():
            self.zoom_window.destroy()
        self.zoom_window = None
        self.zoom_canvas = None
        self.zoom_photo_ref = None

    def open_zoom_window(self) -> None:
        if self.display_image_bgr is None:
            messagebox.showinfo("Thông báo", "Chưa có ảnh để phóng to.")
            return

        if self.zoom_window is not None and self.zoom_window.winfo_exists():
            self.zoom_window.lift()
            self.zoom_window.focus_force()
            return

        self.zoom_window = tk.Toplevel(self.root)
        self.zoom_window.title("Kính lúp ảnh")
        self.zoom_window.geometry("1120x780")
        self.zoom_window.configure(bg="#EEF2F8")
        self.zoom_window.protocol("WM_DELETE_WINDOW", self._close_zoom_window)

        toolbar = tk.Frame(self.zoom_window, bg="#EEF2F8")
        toolbar.pack(fill=tk.X, padx=12, pady=(10, 8))

        ttk.Button(toolbar, text="-", command=self._zoom_out, style="Ghost.TButton").pack(side=tk.LEFT)
        ttk.Button(toolbar, text="+", command=self._zoom_in, style="Ghost.TButton").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="100%", command=self._zoom_reset, style="Ghost.TButton").pack(side=tk.LEFT, padx=(6, 0))
        ttk.Button(toolbar, text="Vừa khung", command=self._zoom_fit, style="Ghost.TButton").pack(
            side=tk.LEFT, padx=(6, 0)
        )

        zoom_label = tk.Label(
            toolbar,
            textvariable=self.zoom_scale_var,
            bg="#EEF2F8",
            fg="#0E2036",
            font=("Segoe UI", 11, "bold"),
        )
        zoom_label.pack(side=tk.LEFT, padx=(10, 0))

        zoom_frame = tk.Frame(self.zoom_window, bg="#EEF2F8")
        zoom_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        v_scroll = tk.Scrollbar(zoom_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = tk.Scrollbar(zoom_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        self.zoom_canvas = tk.Canvas(
            zoom_frame,
            bg="#111827",
            highlightthickness=0,
            xscrollcommand=h_scroll.set,
            yscrollcommand=v_scroll.set,
        )
        self.zoom_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        h_scroll.config(command=self.zoom_canvas.xview)
        v_scroll.config(command=self.zoom_canvas.yview)

        self.zoom_canvas.bind("<MouseWheel>", self._on_zoom_mousewheel)
        self.zoom_canvas.bind("<Button-4>", self._on_zoom_mousewheel)
        self.zoom_canvas.bind("<Button-5>", self._on_zoom_mousewheel)

        self.zoom_window.update_idletasks()
        self._zoom_fit()

    def _setup_video_output_window(
        self,
        frame_shape: Any,
        screen_w: int,
        screen_h: int,
        window_title: Optional[str] = None,
    ) -> None:
        frame_h, frame_w = frame_shape[:2]
        is_vertical = frame_h > frame_w
        title = window_title or self.video_window_title

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        if is_vertical:
            cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

            max_w = int(screen_w * 0.72)
            max_h = int(screen_h * 0.90)
            scale = min(max_w / max(1, frame_w), max_h / max(1, frame_h))
            win_w = max(360, int(frame_w * scale))
            win_h = max(480, int(frame_h * scale))

            cv2.resizeWindow(title, win_w, win_h)
            pos_x = max(0, (screen_w - win_w) // 2)
            pos_y = max(0, (screen_h - win_h) // 2)
            cv2.moveWindow(title, pos_x, pos_y)
        else:
            cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def _build_trace_for_detections(
        self,
        detections: List[Dict[str, Any]],
        vehicle_preds: List[Dict[str, Any]],
        vis_bgr: np.ndarray,
    ) -> None:
        if self.selected_image_bgr is None:
            self.latest_process_trace = None
            return

        steps: List[Dict[str, Any]] = []
        steps.append(
            {
                "title": "Bước 1: Ảnh đầu vào",
                "description": "Ảnh gốc được đọc từ file và đưa vào pipeline.",
                "image": self.selected_image_bgr.copy(),
            }
        )

        det_preview: List[Dict[str, Any]] = []
        for det in detections:
            det_item = dict(det)
            det_item["best_class"] = det.get("det_class_name", "Unknown")
            det_item["best_score"] = float(det.get("det_conf", 0.0))
            det_preview.append(det_item)

        det_vis = draw_vehicle_detections(self.selected_image_bgr, det_preview)
        steps.append(
            {
                "title": "Bước 2: Phát hiện phương tiện",
                "description": f"YOLO phát hiện {len(detections)} khung phương tiện hợp lệ.",
                "image": det_vis,
            }
        )

        max_vehicle_steps = 8
        for i, det in enumerate(vehicle_preds[:max_vehicle_steps], start=1):
            x1, y1, x2, y2 = det["box"]
            crop = self.selected_image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            vn_label = CLASS_NAMES_VN.get(det["best_class"], det["best_class"])
            top_preds = det.get("top_preds", [])
            top_text = ", ".join(
                [
                    f"{CLASS_NAMES_VN.get(cls_name, cls_name)} {score * 100:.1f}%"
                    for cls_name, score in top_preds[:3]
                ]
            )
            desc = (
                f"Cắt theo box ({x1},{y1})-({x2},{y2}), resize 224x224, phân loại VGG16 -> "
                f"{vn_label} {det['best_score'] * 100:.2f}%"
            )
            if top_text:
                desc += f". Top-k: {top_text}."

            steps.append(
                {
                    "title": f"Bước {i + 2}: Xử lý xe #{i}",
                    "description": desc,
                    "image": crop.copy(),
                }
            )

        if len(vehicle_preds) > max_vehicle_steps:
            steps.append(
                {
                    "title": "Bước phụ: Danh sách rút gọn",
                    "description": (
                        f"Có thêm {len(vehicle_preds) - max_vehicle_steps} phương tiện khác, "
                        "không hiển thị chi tiết để tránh quá tải giao diện."
                    ),
                    "image": None,
                }
            )

        steps.append(
            {
                "title": "Bước cuối: Kết quả hiển thị",
                "description": "Ảnh sau cùng với khung vuông và nhãn dự đoán cho từng phương tiện.",
                "image": vis_bgr.copy(),
            }
        )

        self.latest_process_trace = {
            "title": "Quy trình xử lý ảnh (phát hiện nhiều phương tiện)",
            "summary_lines": [
                f"Ảnh: {os.path.basename(self.selected_image_path or 'Không rõ')}",
                f"Số khung phát hiện: {len(detections)}",
                f"Số phương tiện sau phân loại: {len(vehicle_preds)}",
            ],
            "steps": steps,
        }

    def _build_trace_for_multicrop(
        self,
        top_preds: List[Any],
        num_views: int,
    ) -> None:
        if self.selected_image_bgr is None:
            self.latest_process_trace = None
            return

        lines = []
        for i, (cls_name, score) in enumerate(top_preds, start=1):
            vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
            lines.append(f"Top{i}: {vn_label} {score * 100:.2f}%")

        preview = draw_prediction_text(self.selected_image_bgr, lines)
        best_cls, best_score = top_preds[0]
        best_vn = CLASS_NAMES_VN.get(best_cls, best_cls)

        self.latest_process_trace = {
            "title": "Quy trình xử lý ảnh (fallback toàn ảnh)",
            "summary_lines": [
                f"Ảnh: {os.path.basename(self.selected_image_path or 'Không rõ')}",
                "Không phát hiện được box phương tiện bằng detector.",
                f"Dùng chế độ multi-crop với {num_views} vùng ảnh.",
                f"Kết quả tốt nhất: {best_vn} {best_score * 100:.2f}%.",
            ],
            "steps": [
                {
                    "title": "Bước 1: Ảnh đầu vào",
                    "description": "Ảnh gốc được đưa vào chế độ multi-crop.",
                    "image": self.selected_image_bgr.copy(),
                },
                {
                    "title": "Bước 2: Quét nhiều vùng ảnh",
                    "description": (
                        "Mỗi vùng crop được resize 224x224 rồi phân loại. "
                        "Hệ thống tổng hợp xác suất để ra kết quả cuối."
                    ),
                    "image": preview,
                },
            ],
        }

    def _open_process_trace_window(self) -> None:
        trace = self.latest_process_trace
        if trace is None:
            messagebox.showinfo("Thông báo", "Chưa có dữ liệu quy trình xử lý để hiển thị.")
            return

        if self.trace_window is not None and self.trace_window.winfo_exists():
            self.trace_window.destroy()

        self.trace_window = tk.Toplevel(self.root)
        self.trace_window.title(trace.get("title", "Quy trình xử lý ảnh"))
        self.trace_window.geometry("1180x780")
        self.trace_window.configure(bg="#EEF2F8")

        header = tk.Frame(self.trace_window, bg="#EEF2F8")
        header.pack(fill=tk.X, padx=14, pady=(12, 8))

        title_label = tk.Label(
            header,
            text=trace.get("title", "Quy trình xử lý ảnh"),
            bg="#EEF2F8",
            fg="#0E2036",
            font=("Segoe UI", 16, "bold"),
            anchor="w",
        )
        title_label.pack(fill=tk.X)

        summary_lines = trace.get("summary_lines", [])
        if summary_lines:
            summary_label = tk.Label(
                header,
                text="\n".join(summary_lines),
                bg="#EEF2F8",
                fg="#334155",
                font=("Segoe UI", 10),
                justify=tk.LEFT,
                anchor="w",
            )
            summary_label.pack(fill=tk.X, pady=(4, 2))

        notebook = ttk.Notebook(self.trace_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

        self.trace_photo_refs = []
        steps = trace.get("steps", [])
        for i, step in enumerate(steps, start=1):
            tab = tk.Frame(notebook, bg="#F8FAFD")
            notebook.add(tab, text=f"Bước {i}")

            step_title = tk.Label(
                tab,
                text=step.get("title", f"Bước {i}"),
                bg="#F8FAFD",
                fg="#0E2036",
                font=("Segoe UI", 13, "bold"),
                anchor="w",
            )
            step_title.pack(fill=tk.X, padx=12, pady=(12, 6))

            step_desc = tk.Label(
                tab,
                text=step.get("description", ""),
                bg="#F8FAFD",
                fg="#334155",
                font=("Segoe UI", 10),
                justify=tk.LEFT,
                wraplength=1080,
                anchor="w",
            )
            step_desc.pack(fill=tk.X, padx=12, pady=(0, 10))

            image = step.get("image")
            image_frame = tk.Frame(tab, bg="#EAF0F8", relief=tk.SUNKEN, bd=1)
            image_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

            if isinstance(image, np.ndarray):
                photo = self._to_photo_image(image, max_w=1080, max_h=560)
                self.trace_photo_refs.append(photo)
                image_label = tk.Label(image_frame, image=photo, bg="#EAF0F8")
                image_label.pack(fill=tk.BOTH, expand=True)
            else:
                placeholder = tk.Label(
                    image_frame,
                    text="Không có ảnh minh họa cho bước này",
                    bg="#EAF0F8",
                    fg="#64748B",
                    font=("Segoe UI", 11, "italic"),
                )
                placeholder.pack(fill=tk.BOTH, expand=True)

        action_bar = tk.Frame(self.trace_window, bg="#EEF2F8")
        action_bar.pack(fill=tk.X, padx=12, pady=(0, 10))
        close_btn = ttk.Button(action_bar, text="Đóng", command=self.trace_window.destroy, style="Ghost.TButton")
        close_btn.pack(side=tk.RIGHT)

    def _on_status_label_click(self, _event=None) -> None:
        if not self.status_click_enabled:
            return
        self._open_process_trace_window()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Thieu model",
                f"Không tìm thấy model tại: {self.model_path}\nHãy train trước bằng train.py",
            )
            self._set_status("Lỗi: Chưa có model.")
            return

        if not os.path.exists(self.class_map_path):
            messagebox.showerror(
                "Thieu class map",
                f"Không tìm thấy class map tại: {self.class_map_path}",
            )
            self._set_status("Lỗi: Chưa có class map.")
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            class_to_idx = load_class_indices(self.class_map_path)
            self.idx_to_class = {int(v): k for k, v in class_to_idx.items()}
            self._set_status("Đã load model thành công. Hãy chọn ảnh để dự đoán.")
        except Exception as exc:
            messagebox.showerror("Loi load model", str(exc))
            self._set_status("Lỗi khi load model.")

    def choose_image(self) -> None:
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=filetypes)
        if not path:
            return

        image_bgr = read_image_bgr(path)
        if image_bgr is None:
            messagebox.showerror("Lỗi", "Không đọc được file ảnh.")
            return

        self._reset_saved_result_state(clean_temp_video=True)

        self.selected_video_path = None
        self.selected_image_path = path
        self.selected_image_bgr = image_bgr
        self.latest_process_trace = None
        self._show_image(image_bgr)
        self.btn_predict.config(state=tk.NORMAL)
        if self.btn_zoom is not None:
            self.btn_zoom.config(state=tk.NORMAL)
        self._set_status(f"Đã chọn: {os.path.basename(path)}")

    def choose_video(self) -> None:
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.m4v"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Chọn video", filetypes=filetypes)
        if not path:
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được file video.")
            return

        ret, first_frame = cap.read()
        cap.release()

        if not ret or first_frame is None:
            messagebox.showerror("Lỗi", "Không đọc được frame đầu của video.")
            return

        self._reset_saved_result_state(clean_temp_video=True)

        self.selected_image_path = None
        self.selected_video_path = path
        self.selected_image_bgr = first_frame
        self.latest_process_trace = None
        self._show_image(first_frame)
        self.btn_predict.config(state=tk.NORMAL)
        if self.btn_zoom is not None:
            self.btn_zoom.config(state=tk.NORMAL)
        self._set_status(f"Đã chọn video: {os.path.basename(path)}")

    def _show_image(self, image_bgr: np.ndarray) -> None:
        self.display_image_bgr = image_bgr.copy()

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        max_w, max_h = 440, 500
        pil_img.thumbnail((max_w, max_h))

        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.photo_ref, text="")

        if self.btn_zoom is not None:
            self.btn_zoom.config(state=tk.NORMAL)

        if self.zoom_window is not None and self.zoom_window.winfo_exists():
            self._render_zoom_canvas()

    def predict_image(self) -> None:
        if self.model is None or self.idx_to_class is None:
            messagebox.showwarning("Cảnh báo", "Model chưa được load.")
            return

        if self.selected_video_path:
            self.predict_video()
            return

        if self.selected_image_bgr is None:
            messagebox.showwarning("Cảnh báo", "Hãy chọn ảnh trước.")
            return

        self._reset_saved_result_state(clean_temp_video=True)

        try:
            detections = detect_vehicle_boxes(image_bgr=self.selected_image_bgr)
        except ImportError as exc:
            messagebox.showerror(
                "Thiếu thư viện phát hiện",
                f"{exc}\n\nCài thêm bằng lệnh:\n  pip install ultralytics",
            )
            self._set_status("Lỗi: chưa có thư viện phát hiện đối tượng.")
            return

        if detections:
            vehicle_preds = classify_detected_vehicles(
                model=self.model,
                image_bgr=self.selected_image_bgr,
                detections=detections,
                idx_to_class=self.idx_to_class,
                image_size=(224, 224),
                top_k=min(4, len(self.idx_to_class)),
            )

            vis_bgr = draw_vehicle_detections(self.selected_image_bgr, vehicle_preds)
            self._show_image(vis_bgr)

            self.result_main.config(text=f"OUTPUT: phát hiện {len(vehicle_preds)} phương tiện")
            self.top_title.config(text="Danh sách phương tiện phát hiện:")

            lines = []
            for i, det in enumerate(vehicle_preds, start=1):
                x1, y1, x2, y2 = det["box"]
                vn_label = CLASS_NAMES_VN.get(det["best_class"], det["best_class"])
                lines.append(
                    f"{i}. {vn_label:<10} {det['best_score'] * 100:6.2f}% | khung ({x1},{y1})-({x2},{y2})"
                )

            self.top_var.set("\n".join(lines) if lines else "-")
            self._build_trace_for_detections(detections=detections, vehicle_preds=vehicle_preds, vis_bgr=vis_bgr)
            self._remember_image_result(vis_bgr)
            self._set_status(f"Dự đoán thành công: phát hiện {len(vehicle_preds)} phương tiện.", clickable=True)
            return

        # Fallback toàn ảnh nếu detector không tìm thấy phương tiện.
        top_k = len(self.idx_to_class)
        top_preds, num_views = predict_topk_multicrop(
            model=self.model,
            image_bgr=self.selected_image_bgr,
            idx_to_class=self.idx_to_class,
            image_size=(224, 224),
            top_k=top_k,
        )

        self.top_title.config(text=f"Top-{top_k} dự đoán (toàn ảnh):")

        best_class, best_score = top_preds[0]
        vn_name = CLASS_NAMES_VN.get(best_class, best_class)
        self.result_main.config(text=f"OUTPUT: {vn_name} - {best_score * 100:.2f}%")

        lines = []
        for i, (cls_name, score) in enumerate(top_preds, start=1):
            vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
            lines.append(f"{i}. {vn_label:<10} {score * 100:6.2f}%")

        vis_bgr = draw_prediction_text(self.selected_image_bgr, lines)
        self._show_image(vis_bgr)
        self.top_var.set("\n".join(lines))
        self._build_trace_for_multicrop(top_preds=top_preds, num_views=num_views)
        self._remember_image_result(vis_bgr)
        self._set_status(f"Dự đoán thành công (toàn ảnh): quét {num_views} vùng.", clickable=True)

    def predict_video(self) -> None:
        if self.model is None or self.idx_to_class is None:
            messagebox.showwarning("Cảnh báo", "Model chưa được load.")
            return

        if not self.selected_video_path:
            messagebox.showwarning("Cảnh báo", "Hãy chọn video trước.")
            return

        self._reset_saved_result_state(clean_temp_video=True)

        cap = cv2.VideoCapture(self.selected_video_path)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được video đã chọn.")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_ms = int(1000 / fps) if fps and fps > 1 else 30

        window_title = self.video_window_title
        frame_count = 0
        frame_has_vehicle = 0
        total_detected_vehicles = 0
        last_vis: Optional[np.ndarray] = None
        class_counter: Dict[str, int] = {}
        window_initialized = False
        self.root.update_idletasks()
        screen_w = max(1, int(self.root.winfo_screenwidth()))
        screen_h = max(1, int(self.root.winfo_screenheight()))

        os.makedirs("outputs", exist_ok=True)
        video_run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
        temp_output_fast_path = os.path.join(
            "outputs",
            f"_tmp_result_video_fast_{video_run_tag}.mp4",
        )
        temp_output_slow_path = os.path.join(
            "outputs",
            f"_tmp_result_video_slow_{video_run_tag}.mp4",
        )

        writer: Optional[cv2.VideoWriter] = None
        import_error: Optional[Exception] = None
        stopped_early = False
        start_time = time.perf_counter()

        self.video_running = True
        self.video_stop_requested = False
        self._set_video_controls_busy(True)
        self._set_status("Đang phân tích video...", clickable=False)

        try:
            while True:
                if self.video_stop_requested:
                    stopped_early = True
                    break

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
                    # Chế độ video ưu tiên detector để bám sát kiểu camera AI realtime.
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
                else:
                    top_preds, _ = predict_topk_multicrop(
                        model=self.model,
                        image_bgr=frame_for_ai,
                        idx_to_class=self.idx_to_class,
                        image_size=(224, 224),
                        top_k=min(2, len(self.idx_to_class)),
                    )
                    lines = []
                    for i, (cls_name, score) in enumerate(top_preds, start=1):
                        vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
                        lines.append(f"Top{i}: {vn_label} {score * 100:.1f}%")
                    vis_frame = draw_prediction_text(frame_for_ai, lines)

                if not window_initialized:
                    try:
                        self._setup_video_output_window(
                            vis_frame.shape,
                            screen_w,
                            screen_h,
                            window_title=window_title,
                        )
                    except cv2.error:
                        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
                    window_initialized = True

                if writer is None:
                    out_h, out_w = vis_frame.shape[:2]
                    out_fps = float(fps) if fps and fps > 1 else 24.0
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(temp_output_fast_path, fourcc, out_fps, (out_w, out_h))
                    if not writer.isOpened():
                        writer.release()
                        writer = None
                        temp_output_fast_path = ""

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

                if writer is not None:
                    writer.write(vis_frame)

                last_vis = vis_frame
                display_frame = vis_frame
                if window_initialized:
                    try:
                        _, _, win_w, win_h = cv2.getWindowImageRect(window_title)
                        if win_w > 0 and win_h > 0:
                            display_frame = _letterbox_center_frame(vis_frame, win_w, win_h)
                    except cv2.error:
                        pass
                cv2.imshow(window_title, display_frame)

                key = cv2.waitKey(max(1, wait_ms)) & 0xFF
                if key in (ord("q"), 27):
                    stopped_early = True
                    break

        except ImportError as exc:
            import_error = exc
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            try:
                cv2.destroyWindow(window_title)
            except cv2.error:
                pass

            self.video_running = False
            self.video_stop_requested = False
            self._set_video_controls_busy(False)

        if import_error is not None:
            for temp_path in [temp_output_fast_path, temp_output_slow_path]:
                if not temp_path or (not os.path.exists(temp_path)):
                    continue
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            messagebox.showerror(
                "Thiếu thư viện phát hiện",
                f"{import_error}\n\nCài thêm bằng lệnh:\n  pip install ultralytics",
            )
            self._set_status("Lỗi: chưa có thư viện phát hiện đối tượng.", clickable=False)
            return

        if frame_count == 0:
            for temp_path in [temp_output_fast_path, temp_output_slow_path]:
                if not temp_path or (not os.path.exists(temp_path)):
                    continue
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            messagebox.showwarning("Cảnh báo", "Video không có frame hợp lệ để xử lý.")
            return

        if last_vis is not None:
            self._show_image(last_vis)

        avg_vehicle = (total_detected_vehicles / frame_has_vehicle) if frame_has_vehicle > 0 else 0.0

        self.result_main.config(text=f"OUTPUT: video ({frame_count} frame đã xử lý)")
        self.top_title.config(text="Tóm tắt kết quả video:")
        self.top_var.set(
            "\n".join(
                [
                    f"- Tổng frame: {frame_count}",
                    f"- Frame có phát hiện xe: {frame_has_vehicle}",
                    f"- Tổng số xe đã phát hiện: {total_detected_vehicles}",
                    f"- Trung bình xe/frame có phát hiện: {avg_vehicle:.2f}",
                    f"- Xe máy phát hiện: {class_counter.get('Motobikes', 0)}",
                    "- Lưu kết quả sẽ xuất: 1 bản nhanh + 1 bản chậm",
                    "- Mẹo: Nhấn q trong cửa sổ video để dừng sớm",
                ]
            )
        )

        if (
            temp_output_fast_path
            and os.path.exists(temp_output_fast_path)
            and os.path.getsize(temp_output_fast_path) > 0
        ):
            elapsed_sec = max(1e-6, time.perf_counter() - start_time)
            processing_fps = frame_count / elapsed_sec
            slow_fps = max(2.0, min(12.0, processing_fps))

            slow_ready = self._write_video_with_target_fps(
                source_path=temp_output_fast_path,
                target_path=temp_output_slow_path,
                target_fps=slow_fps,
            )

            remembered_slow_path = None
            if slow_ready and os.path.exists(temp_output_slow_path) and os.path.getsize(temp_output_slow_path) > 0:
                remembered_slow_path = temp_output_slow_path

            self._remember_video_result(temp_output_fast_path, remembered_slow_path)
        else:
            self._set_save_button_enabled(False)
            self._set_replay_button_enabled(False)

        self.latest_process_trace = None
        video_name = os.path.basename(self.selected_video_path)
        suffix = " (dừng sớm)" if stopped_early else ""
        self._set_status(
            f"Dự đoán video thành công: {video_name} ({frame_count} frame){suffix}. Bấm 'Xem lại video' để phát lại.",
            clickable=False,
        )

    def clear_view(self) -> None:
        self._reset_saved_result_state(clean_temp_video=True)
        self.selected_image_path = None
        self.selected_image_bgr = None
        self.selected_video_path = None
        self.display_image_bgr = None
        self.photo_ref = None
        self.latest_process_trace = None
        self._close_zoom_window()
        self.image_label.config(image="", text="Chưa có ảnh")
        self.result_main.config(text="OUTPUT: loại phương tiện")
        self.top_var.set("-")
        self._set_status("Đã xóa kết quả. Hãy chọn ảnh mới.")
        self.btn_predict.config(state=tk.DISABLED)
        if self.btn_zoom is not None:
            self.btn_zoom.config(state=tk.DISABLED)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)

    # Duy trì theme phổ biến trên Windows nếu có.
    available = style.theme_names()
    if "vista" in available:
        style.theme_use("vista")

    app = TrafficClassifierGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
