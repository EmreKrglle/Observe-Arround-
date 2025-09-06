import cv2
import numpy as np
import matplotlib.pyplot as plt

class DepthEstimator:
    def __init__(self, focal_length_px=800, baseline_mm=60):
        self.focal_length_px = focal_length_px
        self.baseline_mm = baseline_mm
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    def run(self, camL_index=0, camR_index=1):
        capL = cv2.VideoCapture(camL_index)
        capR = cv2.VideoCapture(camR_index)

        while True:
            retL, frameL = capL.read()
            retR, frameR = capR.read()
            if not retL or not retR:
                print("Kameralardan görüntü alınamadı.")
                break

            grayL = cv2.cvtColor(frameL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(frameR, cv2.COLOR_BGR2GRAY)
            disparity = self.stereo.compute(grayL, grayR).astype(np.float32) / 16.0

            with np.errstate(divide='ignore'):
                depth_map = (self.focal_length_px * self.baseline_mm) / disparity
                depth_map[disparity <= 0] = 0

            plt.imshow(depth_map, cmap="plasma")
            plt.colorbar(label="Mesafe (mm)")
            plt.title("Gerçek Zamanlı Derinlik Haritası")
            plt.pause(0.001)
            plt.clf()

            if cv2.waitKey(1) & 0xFF == 27:
                break

        capL.release()
        capR.release()
        cv2.destroyAllWindows()
