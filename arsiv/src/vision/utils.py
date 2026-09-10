import cv2

# Single place for camera indices so every mode opens the same device
CAMERA_INDEX = 0
STEREO_LEFT_INDEX = 0
STEREO_RIGHT_INDEX = 1

def open_camera(index=CAMERA_INDEX):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {index} açılamadı.")
    return cap

def release_camera(*caps):
    for cap in caps:
        if cap:
            cap.release()
    cv2.destroyAllWindows()
