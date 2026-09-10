from ultralytics import YOLO
import cv2
import time

class ObjectDetector:
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model = YOLO(model_path)

    def _segment_location(self, cw, segment, w):
        if w > cw > w - segment:
            return "right"
        elif w - segment > cw > w - 2 * segment:
            return "right-mid"
        elif w - 2 * segment > cw > w - 3 * segment:
            return "mid"
        elif w - 3 * segment > cw > w - 4 * segment:
            return "left-mid"
        elif w - 4 * segment > cw > w - 5 * segment:
            return "left"
        else:
            return None

    def observe(self, cam_index=1, conf_threshold=0.75, update_callback=None):
        cap = cv2.VideoCapture(cam_index)
        start_time = time.time()
        detected_objects = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            segment = w // 5

            results = self.model.track(
                source=frame,
                conf=0.70,
                retina_masks=True,
                show_boxes=False,
                verbose=False,
                device="cpu"
            )
            annotated_frame = results[0].plot()

            if results[0].boxes.id is not None:
                for cls, conf, obj_id in zip(results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.id):
                    if conf > conf_threshold:
                        name = self.model.names[int(cls)]
                        box = results[0].boxes.xyxy[0].tolist()
                        x1, y1, x2, y2 = box
                        center = (x1 + x2) / 2, (y1 + y2) / 2
                        location = self._segment_location(center[0], segment, w)

                        if location:
                            detected_objects.append((name, location, obj_id))

                if update_callback:
                    update_callback(detected_objects)

                current_time = time.time()
                if current_time - start_time >= 1.0:
                    for name, loc, obj_id in detected_objects:
                        print(f"id:{obj_id}\tobject:{name}\tlocation:{loc}")
                    detected_objects.clear()
                    start_time = current_time

            cv2.imshow("YOLOv8 + Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
