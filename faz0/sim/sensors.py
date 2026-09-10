"""
Sanal ToF sensorleri.

!!! DIKKAT - BU MODEL GERCEGINDEN IYIMSERDIR !!!
================================================
Buradaki sensor modeli su gercek dunya problemlerini TASIMAZ:

  * Islak zemin / su birikintisi -> ayna gibi yansitir, "cevap yok" verir
  * Koyu renkli nesneler        -> siyah mont menzili ~4'e boler
  * Cam ve seffaf yuzeyler      -> neredeyse gorunmezdir
  * Lense konan su damlasi      -> sensor kalici olarak "2 cm'de nesne var" der
  * Sensor kiri, parmak izi, bugulanma

Yani simulasyonda %100 basari almaniz, sahada calisacagi anlamina GELMEZ.
Simulator geometriyi ve karar mantigini dogrular; sensor fizigini degil.
Fizigi ancak gercek donanimla, gercek disarida olcebilirsiniz.

Gurultuyu ve menzili config.py'den agirlastirip sistemin ne kadar
dayanikli oldugunu gorebilirsiniz - bunu yapmanizi tavsiye ederim.
"""

import math
import random

import config
from .geometry import ray_circle, ray_segment


class Reading:
    """Tek bir sensorun tek bir andaki olcumu."""

    __slots__ = ("index", "angle", "distance", "valid")

    def __init__(self, index, angle, distance, valid):
        self.index = index
        self.angle = angle          # govde cercevesinde, radyan
        self.distance = distance    # metre; valid False ise anlamsiz
        self.valid = valid          # False = "cevap yok"


class ToFArray:
    """Govde etrafina yerlestirilmis sanal ToF sensorleri dizisi."""

    def __init__(self, angles_rad=None, rng=None):
        self.angles = list(angles_rad if angles_rad is not None
                           else config.SENSOR_ANGLES_RAD)
        self.rng = rng or random.Random()
        self.last_readings = []

    # -- ic yardimcilar ----------------------------------------------------

    def _cast_one_ray(self, ox, oy, ang_world, world, max_range):
        """Tek isin at, en yakin carpma mesafesini dondur (yoksa None)."""
        dx = math.cos(ang_world)
        dy = math.sin(ang_world)
        best = None

        for w in world.walls:
            t = ray_segment(ox, oy, dx, dy, w.ax, w.ay, w.bx, w.by)
            if t is not None and (best is None or t < best):
                best = t

        for p in world.pillars:
            t = ray_circle(ox, oy, dx, dy, p.x, p.y, p.r)
            if t is not None and (best is None or t < best):
                best = t

        for m in world.movers:
            t = ray_circle(ox, oy, dx, dy, m.x, m.y, m.r)
            if t is not None and (best is None or t < best):
                best = t

        if best is None or best > max_range:
            return None
        return best

    # -- ana arayuz --------------------------------------------------------

    def sense(self, x, y, heading, world, ambient_light=None):
        """Tum sensorleri oku.

        x, y, heading : kullanicinin dunyadaki konumu ve yonu
        world         : World nesnesi
        ambient_light : 0..1; None ise config'teki deger kullanilir

        Donen: Reading listesi
        """
        if ambient_light is None:
            ambient_light = config.AMBIENT_LIGHT
        max_range = config.sensor_max_range(ambient_light)

        fov = math.radians(config.SENSOR_FOV_DEG)
        n_rays = max(1, config.SENSOR_RAYS)

        readings = []
        for i, rel_ang in enumerate(self.angles):
            # Koniyi birden fazla isinla ornekle, en YAKIN sonucu al.
            # Gercek ToF de konisindeki en yakin yuzeyi raporlar.
            nearest = None
            for k in range(n_rays):
                if n_rays == 1:
                    offset = 0.0
                else:
                    offset = -fov / 2.0 + fov * k / (n_rays - 1)
                ang_world = heading + rel_ang + offset
                t = self._cast_one_ray(x, y, ang_world, world, max_range)
                if t is not None and (nearest is None or t < nearest):
                    nearest = t

            if nearest is None:
                readings.append(Reading(i, rel_ang, max_range, False))
                continue

            # Gurultu ekle
            sigma = nearest * config.SENSOR_NOISE_RATIO
            noisy = nearest + self.rng.gauss(0.0, sigma) if sigma > 0 else nearest

            # Rastgele okuma kaybi
            if self.rng.random() < config.SENSOR_DROPOUT_PROB:
                readings.append(Reading(i, rel_ang, max_range, False))
                continue

            if noisy < config.SENSOR_MIN_RANGE_M:
                noisy = config.SENSOR_MIN_RANGE_M
            if noisy > max_range:
                readings.append(Reading(i, rel_ang, max_range, False))
                continue

            readings.append(Reading(i, rel_ang, noisy, True))

        self.last_readings = readings
        return readings


def readings_to_obstacles(readings):
    """Sensor okumalarini (kerteriz, mesafe) ciftlerine cevirir.

    Haptik katmani sensorlerin kac tane oldugunu ya da nerede
    durdugunu bilmez - sadece "su yonde su mesafede bir sey var"
    listesi alir. Gercek sistemde radar ve kamera da bu listeye
    ayni formatta katki yapar.
    """
    return [(r.angle, r.distance) for r in readings if r.valid]
