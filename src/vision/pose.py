from ultralytics import YOLO
import cv2
import time

class PoseEstimator:
    def __init__(self, model_path="yolov8n-pose.pt"):
        self.model = YOLO(model_path)

    def run(self, cam_index=1):
        cap = cv2.VideoCapture(cam_index)
        prev_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = self.model.predict(source=frame, task="pose", conf=0.5, device="cpu")
            annotated = result[0].plot()

            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time
            cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Pose Estimation", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
