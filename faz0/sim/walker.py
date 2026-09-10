"""
Yuruyen kullanici modeli.

Onemli detay: YURUYUS SALINIMI.
Gercek hayatta govde her adimda saga sola hafifce doner. Bu, govdeye
bagli sensorlerin nereye baktigini surekli degistirir ve yalanci
alarmlarin baslica sebebidir.

config.GAIT_SWAY_DEG = 0 yaparsaniz simulasyonda sorun kaybolur -
gercek dunyada kaybolmaz. Sistemin bu salinima dayanikli oldugunu
gormek icin salinimi ARTIRIP test edin.
"""

import math

import config


class Walker:
    def __init__(self, x=0.0, y=0.0, heading_deg=0.0):
        self.x = x
        self.y = y
        self.heading = math.radians(heading_deg)   # niyet edilen yon
        self._gait_phase = 0.0
        self.moving = False

    @property
    def sensed_heading(self):
        """Sensorlerin GERCEKTE baktigi yon.

        Niyet edilen yon + yuruyus salinimi. Sistem bunu bilmez -
        gercek hayatta da bilmez (IMU olmadan).
        """
        sway = math.radians(config.GAIT_SWAY_DEG) * math.sin(self._gait_phase)
        return self.heading + sway

    def update(self, dt, forward=0.0, turn=0.0, strafe=0.0):
        """
        forward : -1..1  (ileri / geri)
        turn    : -1..1  (sol / sag donus)
        strafe  : -1..1  (yana adim)
        """
        self.heading += math.radians(config.TURN_SPEED_DPS) * turn * dt

        vx = vy = 0.0
        if forward:
            vx += math.cos(self.heading) * config.WALK_SPEED_MPS * forward
            vy += math.sin(self.heading) * config.WALK_SPEED_MPS * forward
        if strafe:
            # Sola adim = heading + 90 derece
            left = self.heading + math.pi / 2.0
            vx += math.cos(left) * config.WALK_SPEED_MPS * 0.6 * strafe
            vy += math.sin(left) * config.WALK_SPEED_MPS * 0.6 * strafe

        self.moving = (vx * vx + vy * vy) > 1e-9
        self.x += vx * dt
        self.y += vy * dt

        # Salinim sadece yururken olur, dururken degil
        if self.moving:
            self._gait_phase += 2.0 * math.pi * config.GAIT_FREQ_HZ * dt
            self._gait_phase %= 2.0 * math.pi

    def reset(self, x, y, heading_deg):
        self.x, self.y = x, y
        self.heading = math.radians(heading_deg)
        self._gait_phase = 0.0
        self.moving = False
