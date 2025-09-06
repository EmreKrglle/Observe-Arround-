import cv2

def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Kamera {index} açılamadı.")
    return cap

def release_camera(cap):
    if cap:
        cap.release()
    cv2.destroyAllWindows()
