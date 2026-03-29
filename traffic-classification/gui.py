import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

from utils import CLASS_NAMES_VN, load_class_indices, predict_topk, read_image_bgr


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
        self.photo_ref = None

        self._configure_styles()
        self._build_ui()
        self._load_model()

    def _configure_styles(self) -> None:
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        self.style.configure("Title.TLabel", font=("Segoe UI", 21, "bold"), foreground="#0E2036", background="#EEF2F8")
        self.style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#4A5A73", background="#EEF2F8")
        self.style.configure("CardTitle.TLabel", font=("Segoe UI", 12, "bold"), foreground="#1D2E45", background="#F8FAFD")
        self.style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"), foreground="#0B5ED7", background="#EEF2F8")
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

        self.btn_predict = ttk.Button(
            controls,
            text="Dự đoán",
            command=self.predict_image,
            state=tk.DISABLED,
            style="Primary.TButton",
        )
        self.btn_predict.pack(side=tk.LEFT, padx=8)

        self.btn_clear = ttk.Button(controls, text="Xóa", command=self.clear_view, style="Ghost.TButton")
        self.btn_clear.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Sẵn sàng. Hãy chọn ảnh để dự đoán.")
        status_label = ttk.Label(container, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(anchor=tk.W, pady=(4, 10))

        content = tk.Frame(container, bg="#D9DEE7")
        content.pack(fill=tk.BOTH, expand=True)

        # Left 3D card (input image).
        left_shadow = tk.Frame(content, bg="#B6BFCE")
        left_shadow.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        left = tk.Frame(left_shadow, bg="#F8FAFD", relief=tk.RAISED, bd=2)
        left.pack(fill=tk.BOTH, expand=True, padx=(0, 2), pady=(0, 2))

        left_title = ttk.Label(left, text="Ảnh đầu vào", style="CardTitle.TLabel")
        left_title.pack(anchor=tk.W, padx=12, pady=(10, 8))

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

        self.top_title = ttk.Label(right, text="Top-4 dự đoán:", style="TopTitle.TLabel")
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

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            messagebox.showerror(
                "Thieu model",
                f"Không tìm thấy model tại: {self.model_path}\nHãy train trước bằng train.py",
            )
            self.status_var.set("Lỗi: Chưa có model.")
            return

        if not os.path.exists(self.class_map_path):
            messagebox.showerror(
                "Thieu class map",
                f"Không tìm thấy class map tại: {self.class_map_path}",
            )
            self.status_var.set("Lỗi: Chưa có class map.")
            return

        try:
            self.model = tf.keras.models.load_model(self.model_path)
            class_to_idx = load_class_indices(self.class_map_path)
            self.idx_to_class = {int(v): k for k, v in class_to_idx.items()}
            self.status_var.set("Đã load model thành công. Hãy chọn ảnh để dự đoán.")
        except Exception as exc:
            messagebox.showerror("Loi load model", str(exc))
            self.status_var.set("Lỗi khi load model.")

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

        self.selected_image_path = path
        self.selected_image_bgr = image_bgr
        self._show_image(image_bgr)
        self.btn_predict.config(state=tk.NORMAL)
        self.status_var.set(f"Đã chọn: {os.path.basename(path)}")

    def _show_image(self, image_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        max_w, max_h = 440, 500
        pil_img.thumbnail((max_w, max_h))

        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.image_label.config(image=self.photo_ref, text="")

    def predict_image(self) -> None:
        if self.model is None or self.idx_to_class is None:
            messagebox.showwarning("Cảnh báo", "Model chưa được load.")
            return

        if self.selected_image_bgr is None:
            messagebox.showwarning("Cảnh báo", "Hãy chọn ảnh trước.")
            return

        top_k = len(self.idx_to_class)
        top_preds = predict_topk(
            model=self.model,
            image_bgr=self.selected_image_bgr,
            idx_to_class=self.idx_to_class,
            image_size=(224, 224),
            top_k=top_k,
        )

        self.top_title.config(text=f"Top-{top_k} dự đoán:")

        best_class, best_score = top_preds[0]
        vn_name = CLASS_NAMES_VN.get(best_class, best_class)
        self.result_main.config(text=f"OUTPUT: {vn_name} - {best_score * 100:.2f}%")

        lines = []
        for i, (cls_name, score) in enumerate(top_preds, start=1):
            vn_label = CLASS_NAMES_VN.get(cls_name, cls_name)
            lines.append(f"{i}. {vn_label:<10} {score * 100:6.2f}%")

        self.top_var.set("\n".join(lines))
        self.status_var.set("Dự đoán thành công.")

    def clear_view(self) -> None:
        self.selected_image_path = None
        self.selected_image_bgr = None
        self.photo_ref = None
        self.image_label.config(image="", text="Chưa có ảnh")
        self.result_main.config(text="OUTPUT: loại phương tiện")
        self.top_var.set("-")
        self.status_var.set("Đã xóa kết quả. Hãy chọn ảnh mới.")
        self.btn_predict.config(state=tk.DISABLED)


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
