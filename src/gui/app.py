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

        self.detector = ObjectDetector()
        self.pose = PoseEstimator()
        self.depth = DepthEstimator()

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
        if not objects:
            self.direction_label.config(text="")
        else:
            text = ", ".join([f"{name} at {loc}" for name, loc, _ in objects])
            self.direction_label.config(text=text)
        self.root.update_idletasks()

    def start_detection(self):
        self.detector.observe(update_callback=self.update_direction)

    def start_pose(self):
        self.pose.run()

    def start_depth(self):
        self.depth.run()

    def quit_app(self):
        self.root.destroy()

    def run(self):
        self.root.mainloop()
