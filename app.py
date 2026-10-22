
from __future__ import annotations

import ctypes
import math
import time
import tkinter as tk
from tkinter import messagebox

import cv2
import numpy as np
from PIL import ImageGrab


# =========================================================
# Windows screen helpers
# =========================================================

user32 = ctypes.windll.user32


def cursor_position():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    p = POINT()
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def select_region():
    """Drag a rectangle around the game viewport."""
    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.30)
    root.configure(bg="black")

    canvas = tk.Canvas(
        root, bg="black", cursor="cross",
        highlightthickness=0
    )
    canvas.pack(fill="both", expand=True)

    state = {"start": None, "rect": None, "result": None}

    tk.Label(
        root,
        text="Drag quanh vùng game | ESC để hủy",
        fg="white", bg="black", font=("Arial", 16)
    ).place(x=20, y=20)

    def down(event):
        state["start"] = (event.x_root, event.y_root)
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#00ff66", width=3
        )

    def drag(event):
        if state["start"] is None:
            return
        x0, y0 = state["start"]
        canvas.coords(
            state["rect"],
            x0, y0, event.x_root, event.y_root
        )

    def up(event):
        if state["start"] is None:
            return
        x0, y0 = state["start"]
        x1, y1 = event.x_root, event.y_root
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))

        if right - left >= 200 and bottom - top >= 150:
            state["result"] = (left, top, right, bottom)
        root.destroy()

    canvas.bind("<ButtonPress-1>", down)
    canvas.bind("<B1-Motion>", drag)
    canvas.bind("<ButtonRelease-1>", up)
    root.bind("<Escape>", lambda _e: root.destroy())

    root.mainloop()
    return state["result"]


# =========================================================
# Vision
# =========================================================

def capture(region):
    left, top, right, bottom = region
    img = ImageGrab.grab(bbox=(left, top, right, bottom))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def detect_player(frame, template, previous=None):
    """Return the matched bounding box and confidence."""
    if template is None:
        return None, 0.0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    th, tw = template.shape

    # Search locally after the first detection to reduce false matches.
    if previous is not None:
        px, py, pw, ph = previous
        margin = 100
        x0 = max(0, px - margin)
        y0 = max(0, py - margin)
        x1 = min(gray.shape[1], px + pw + margin)
        y1 = min(gray.shape[0], py + ph + margin)
        search = gray[y0:y1, x0:x1]

        if search.shape[0] >= th and search.shape[1] >= tw:
            result = cv2.matchTemplate(
                search, template, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, loc = cv2.minMaxLoc(result)
            box = (x0 + loc[0], y0 + loc[1], tw, th)
            return (box if score >= 0.60 else None), score

    result = cv2.matchTemplate(
        gray, template, cv2.TM_CCOEFF_NORMED
    )
    _, score, _, loc = cv2.minMaxLoc(result)
    box = (loc[0], loc[1], tw, th)
    return (box if score >= 0.60 else None), score


def detect_platforms(frame):
    """Approximate horizontal surfaces from image edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    kernel_width = max(15, frame.shape[1] // 70)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernel_width, 1)
    )
    horizontal = cv2.morphologyEx(
        edges, cv2.MORPH_OPEN, kernel
    )

    contours, _ = cv2.findContours(
        horizontal, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    platforms = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if w >= kernel_width and h <= 8:
            platforms.append((x, y, x + w))

    return sorted(platforms, key=lambda p: (p[1], p[0]))


# =========================================================
# Hypothetical trajectory model
# =========================================================

def simulate_jump(
    start_x, start_y, angle_deg,
    speed, gravity, platforms,
    player_width, frame_width, frame_height
):
    """
    Coordinates use screen convention:
    x increases right; y increases downward.

    This is a tunable ballistic approximation, not the
    confirmed internal physics of Jump King.
    """
    angle = math.radians(angle_deg)
    vx = speed * math.cos(angle)
    vy = -speed * math.sin(angle)

    points = []
    landing = None
    dt = 0.025
    t = 0.0
    max_time = 3.0

    previous_x, previous_y = start_x, start_y

    while t <= max_time:
        x = start_x + vx * t
        y = start_y + vy * t + 0.5 * gravity * t * t

        if not (
            0 <= x < frame_width
            and 0 <= y < frame_height
        ):
            break

        points.append((int(x), int(y)))

        # Detect crossing a platform top while falling.
        if t > 0 and y >= previous_y:
            for left, top, right in platforms:
                crossed_top = previous_y <= top <= y
                overlaps = (
                    x + player_width / 2 >= left
                    and x - player_width / 2 <= right
                )

                if crossed_top and overlaps:
                    landing = (int(x), int(top))
                    break

        if landing is not None:
            break

        previous_x, previous_y = x, y
        t += dt

    return points, landing


# =========================================================
# Realtime overlay and controls
# =========================================================


class JumpKingAssistant:
    def __init__(self, region):
        self.region = region
        self.template = None
        self.previous_box = None
        self.last_capture = None
        self.running = True
        self.last_tick = 0.0

        # FIX: Tạo cửa sổ Tk trước các biến Tkinter.
        self.root = tk.Tk()
        self.root.title("Jump King Trajectory Assistant")
        self.root.geometry("390x430+30+30")
        self.root.attributes("-topmost", True)

        # FIX: Gắn rõ master để tránh lỗi default root.
        self.angle = tk.DoubleVar(master=self.root, value=55)
        self.speed = tk.DoubleVar(master=self.root, value=300)
        self.gravity = tk.DoubleVar(master=self.root, value=900)

        tk.Label(
            self.root,
            text="JUMP KING — REALTIME ASSISTANT",
            font=("Segoe UI", 11, "bold")
        ).pack(pady=8)

        self.status = tk.StringVar(
            master=self.root,
            value="Đang khởi tạo..."
        )
        tk.Label(
            self.root,
            textvariable=self.status,
            wraplength=360,
            justify="left"
        ).pack(padx=8, pady=5)

        self.add_slider("Góc giả lập", self.angle, 10, 85, "°")
        self.add_slider("Tốc độ giả lập", self.speed, 50, 700, "")
        self.add_slider("Trọng lực giả lập", self.gravity, 200, 1800, "")

        tk.Button(
            self.root,
            text="Lấy mẫu nhân vật (F6)",
            command=self.calibrate
        ).pack(fill="x", padx=12, pady=4)

        tk.Button(
            self.root,
            text="Đọc lại mẫu nhân vật",
            command=self.reset_template
        ).pack(fill="x", padx=12, pady=4)

        tk.Button(
            self.root,
            text="Thoát",
            command=self.stop
        ).pack(fill="x", padx=12, pady=4)

        self.root.bind("<F6>", lambda _e: self.calibrate())
        self.root.protocol("WM_DELETE_WINDOW", self.stop)

        # Overlay trong suốt phủ đúng vùng game.
        left, top, right, bottom = region
        self.width = right - left
        self.height = bottom - top

        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.geometry(
            f"{self.width}x{self.height}+{left}+{top}"
        )
        self.overlay.configure(bg="magenta")

        try:
            self.overlay.attributes("-transparentcolor", "magenta")
        except tk.TclError:
            pass

        self.canvas = tk.Canvas(
            self.overlay,
            width=self.width,
            height=self.height,
            bg="magenta",
            highlightthickness=0
        )
        self.canvas.pack()

        # Cố gắng loại overlay khỏi ảnh chụp trên Windows hỗ trợ.
        self.capture_exclusion = False
        try:
            self.overlay.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(
                self.overlay.winfo_id()
            )
            self.capture_exclusion = bool(
                ctypes.windll.user32.SetWindowDisplayAffinity(
                    hwnd, 0x11
                )
            )
        except Exception:
            pass

        self.root.after(100, self.tick)

    def add_slider(self, label, variable, low, high, suffix):
        tk.Label(self.root, text=label).pack(
            anchor="w", padx=12
        )
        tk.Scale(
            self.root,
            variable=variable,
            from_=low,
            to=high,
            orient="horizontal",
            resolution=1,
            length=340
        ).pack(padx=8)

    def reset_template(self):
        self.template = None
        self.previous_box = None
        self.status.set(
            "Đã xóa mẫu. Đặt chuột lên giữa nhân vật rồi nhấn F6."
        )

    def calibrate(self):
        if self.last_capture is None:
            self.status.set("Chưa có ảnh màn hình để lấy mẫu.")
            return

        sx, sy = cursor_position()
        left, top, right, bottom = self.region
        x, y = sx - left, sy - top

        frame = self.last_capture
        h, w = frame.shape[:2]

        if not (0 <= x < w and 0 <= y < h):
            self.status.set("Chuột phải nằm trong vùng game.")
            return

        # Mẫu nhỏ, cố định kích thước để template matching.
        crop_w = min(40, w)
        crop_h = min(54, h)

        x0 = max(0, min(w - crop_w, int(x - crop_w / 2)))
        y0 = max(0, min(h - crop_h, int(y - crop_h / 2)))

        crop = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        if crop.size == 0:
            self.status.set("Không lấy được mẫu; thử lại.")
            return

        self.template = cv2.cvtColor(
            crop, cv2.COLOR_BGR2GRAY
        )
        self.previous_box = None

        self.status.set(
            "Đã lấy mẫu nhân vật. Nếu khung nhận diện sai, "
            "thử lấy mẫu lại khi nhân vật đứng yên."
        )

    def tick(self):
        if not self.running:
            return

        try:
            # Nếu Windows không loại overlay khỏi ảnh chụp,
            # tạm ẩn overlay trước khi đọc màn hình.
            if not self.capture_exclusion:
                self.overlay.withdraw()
                self.root.update_idletasks()

            frame = capture(self.region)
            self.last_capture = frame.copy()

            if not self.capture_exclusion:
                self.overlay.deiconify()
                self.overlay.lift()

            self.canvas.delete("all")

            if frame is None or frame.size == 0:
                self.status.set("Không đọc được ảnh game.")
                return

            frame_h, frame_w = frame.shape[:2]
            platforms = detect_platforms(frame)

            # Vẽ bề mặt nền phát hiện được.
            for x1, py, x2 in platforms:
                self.canvas.create_line(
                    x1, py, x2, py,
                    fill="#66AAFF",
                    width=2
                )

            box = None
            confidence = 0.0

            if self.template is not None:
                box, confidence = detect_player(
                    frame,
                    self.template,
                    self.previous_box
                )

            if box is not None:
                self.previous_box = box
                x, y, w, h = box
                cx = x + w / 2
                feet_y = y + h

                self.canvas.create_rectangle(
                    x, y, x + w, y + h,
                    outline="#00FFFF",
                    width=2
                )

                # Quỹ đạo mô phỏng. Các tham số phải hiệu chỉnh
                # theo chuyển động thực tế của Jump King.
                points, landing = simulate_jump(
                    cx,
                    feet_y,
                    self.angle.get(),
                    self.speed.get(),
                    self.gravity.get(),
                    platforms,
                    w,
                    frame_w,
                    frame_h
                )

                if len(points) >= 2:
                    self.canvas.create_line(
                        *[
                            coordinate
                            for point in points
                            for coordinate in point
                        ],
                        fill="#00FF66",
                        width=2,
                        smooth=True
                    )

                    for px, py in points[::5]:
                        self.canvas.create_oval(
                            px - 2, py - 2,
                            px + 2, py + 2,
                            fill="#00FF66",
                            outline=""
                        )

                if landing is not None:
                    lx, ly = landing

                    self.canvas.create_oval(
                        lx - 7, ly - 7,
                        lx + 7, ly + 7,
                        outline="#FFFF00",
                        width=3
                    )
                    self.canvas.create_line(
                        lx - 16, ly, lx + 16, ly,
                        fill="#FFFF00",
                        width=2
                    )
                    self.canvas.create_rectangle(
                        lx - w / 2, ly - 4,
                        lx + w / 2, ly + 4,
                        outline="#FFFF00",
                        width=2
                    )

                landing_text = (
                    f" | Điểm rơi: ({landing[0]}, {landing[1]})"
                    if landing else " | Chưa tìm thấy điểm rơi"
                )

                self.status.set(
                    f"Nhận diện: {confidence:.2f} | "
                    f"Góc: {self.angle.get():.0f}° | "
                    f"Tốc độ: {self.speed.get():.0f} | "
                    f"Trọng lực: {self.gravity.get():.0f}"
                    + landing_text
                )

            elif self.template is None:
                self.status.set(
                    "Đang đọc màn hình. Đặt chuột vào nhân vật "
                    "rồi nhấn F6 để lấy mẫu."
                )
            else:
                self.status.set(
                    f"Chưa nhận diện được nhân vật "
                    f"(điểm khớp: {confidence:.2f}). "
                    "Thử F6 lấy mẫu lại."
                )

        except Exception as exc:
            self.status.set(f"Lỗi đọc màn hình: {exc}")

        finally:
            if self.running:
                self.root.after(50, self.tick)

    def stop(self):
        self.running = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    if not hasattr(ctypes, "windll"):
        raise SystemExit("Chương trình này dành cho Windows.")

    print("Khoanh vùng cửa sổ Jump King.")
    region = select_region()

    if region:
        JumpKingAssistant(region).run()
    else:
        print("Đã hủy.")
