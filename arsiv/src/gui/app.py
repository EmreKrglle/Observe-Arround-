import threading
import tkinter as tk
from vision.detector import ObjectDetector
from vision.pose import PoseEstimator
from vision.depth import DepthEstimator

class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SEE Project")
        self.root.geometry("1000x800")
        self.root.configure(bg="black")

        # Models load on first use (inside the worker thread) to keep startup fast
        self.detector = None
        self.pose = None
        self.depth = None
        self.worker = None

        self.direction_label = tk.Label(
            self.root, text="", font=("Arial", 36, "bold"),
            fg="white", bg="black", height=1
        )
        self.direction_label.pack(side="bottom", fill=tk.X, pady=40)

        label = tk.Label(
            self.root, text="Welcome to SEE Interface",
            font=("Arial", 28, "bold"), fg="purple", bg="black"
        )
        label.pack(pady=20)

        self._create_buttons()

    def _create_buttons(self):
        btn_style = {
            "font": ("Arial", 14),
            "bg": "white",
            "fg": "black",
            "width": 30,
            "height": 2,
            "relief": tk.RAISED,
            "bd": 3
        }

        tk.Button(self.root, text="Observe Around",
                  command=self.start_detection, **btn_style).pack(pady=5)
        tk.Button(self.root, text="Pose Estimation",
                  command=self.start_pose, **btn_style).pack(pady=5)
        tk.Button(self.root, text="Depth Map",
                  command=self.start_depth, **btn_style).pack(pady=5)
        tk.Button(self.root, text="Quit",
                  command=self.quit_app, **btn_style).pack(pady=5)

    def update_direction(self, objects):
        # Called from the worker thread; Tkinter is not thread-safe, so hop to the main loop
        text = ", ".join(f"{name} at {loc}" for name, loc, _ in objects)
        self.root.after(0, self.direction_label.config, {"text": text})

    def _set_status(self, text):
        self.root.after(0, self.direction_label.config, {"text": text})

    def _run_in_background(self, target):
        # One camera task at a time; they would fight over the same camera otherwise
        if self.worker and self.worker.is_alive():
            return

        def task():
            try:
                target()
                self._set_status("")
            except Exception as e:
                self._set_status(str(e))

        self.worker = threading.Thread(target=task, daemon=True)
        self.worker.start()

    def start_detection(self):
        def task():
            if self.detector is None:
                self._set_status("Model yükleniyor...")
                self.detector = ObjectDetector()
            self.detector.observe(update_callback=self.update_direction)
        self._run_in_background(task)

    def start_pose(self):
        def task():
            if self.pose is None:
                self._set_status("Model yükleniyor...")
                self.pose = PoseEstimator()
            self._set_status("")
            self.pose.run()
        self._run_in_background(task)

    def start_depth(self):
        def task():
            if self.depth is None:
                self.depth = DepthEstimator()
            self.depth.run()
        self._run_in_background(task)

    def quit_app(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()
