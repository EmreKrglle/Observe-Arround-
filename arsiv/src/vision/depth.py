import cv2
import numpy as np
from vision.utils import STEREO_LEFT_INDEX, STEREO_RIGHT_INDEX, open_camera, release_camera

class DepthEstimator:
    def __init__(self, focal_length_px=800, baseline_mm=60):
        self.focal_length_px = focal_length_px
        self.baseline_mm = baseline_mm
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)

    def run(self, camL_index=STEREO_LEFT_INDEX, camR_index=STEREO_RIGHT_INDEX):
        capL = open_camera(camL_index)
        try:
            capR = open_camera(camR_index)
        except RuntimeError:
            release_camera(capL)
            raise

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

            # cv2 window instead of matplotlib: pyplot cannot run outside the main thread
            depth_vis = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_PLASMA)
            cv2.imshow("Gercek Zamanli Derinlik Haritasi", depth_vis)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        release_camera(capL, capR)
