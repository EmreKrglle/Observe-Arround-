"""
Basari olcumu.

"Calisiyor" demek yetmez - sayiyla ifade edilmeli. Bunlar Faz 0'in
basari kriterleri:

    carpisma        : 0 olmali
    yalanci alarm   : dakikada 1'in ALTINDA olmali
    kacirilan engel : %5'in altinda olmali

Yalanci alarm en onemlisidir. Surekli bosuna titreyen bir bant,
hic titremeyen bir banttan daha cabuk cope atilir.
"""

import config


class Scorer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self.distance = 0.0
        self.collisions = 0
        self.false_alarms = 0
        self.missed_frames = 0
        self.alert_frames = 0
        self.min_clearance = float("inf")
        self.goal_reached = False
        self.goal_time = None
        self._in_collision = False
        self._in_false_alarm = False
        self._last_xy = None

    def update(self, dt, walker, world, obstacles):
        """obstacles: sistemin gordugu [(kerteriz, mesafe), ...] listesi."""
        self.elapsed += dt

        # Yol uzunlugu
        if self._last_xy is not None:
            dx = walker.x - self._last_xy[0]
            dy = walker.y - self._last_xy[1]
            self.distance += (dx * dx + dy * dy) ** 0.5
        self._last_xy = (walker.x, walker.y)

        clearance = world.min_clearance(walker.x, walker.y)
        if clearance < self.min_clearance:
            self.min_clearance = clearance

        # -- carpisma (yukselen kenar: her carpma bir kez sayilir) --------
        colliding = clearance <= config.BODY_RADIUS_M
        if colliding and not self._in_collision:
            self.collisions += 1
        self._in_collision = colliding

        # -- sistem uyari veriyor mu ----------------------------------------
        # Motorun o anki acik/kapali durumuna degil, sistemin KARARINA bakilir.
        # Engel kalibi nabiz seklindedir; nabzin kapali oldugu anlar
        # "sessiz bant" degildir. Motor durumuna bakmak her nabiz arasini
        # kacirma, her nabzi yeni bir yalanci alarm diye sayar.
        alerting = any(d <= config.ALERT_DISTANCE_M for _, d in obstacles)

        # -- yalanci alarm -------------------------------------------------
        # Ortada engel yokken uyari veriyorsa yalanci alarmdir.
        nothing_near = clearance > config.ALERT_DISTANCE_M
        false_now = alerting and nothing_near
        if false_now and not self._in_false_alarm:
            self.false_alarms += 1
        self._in_false_alarm = false_now

        # -- kacirilan engel -----------------------------------------------
        # Engel uyari mesafesinde ama bant sessiz.
        if not nothing_near:
            self.alert_frames += 1
            if not alerting:
                self.missed_frames += 1

        if not self.goal_reached and world.reached_goal(walker.x, walker.y):
            self.goal_reached = True
            self.goal_time = self.elapsed

    # -- turetilmis olcutler ------------------------------------------------

    @property
    def false_alarms_per_minute(self):
        if self.elapsed <= 0.0:
            return 0.0
        return self.false_alarms / (self.elapsed / 60.0)

    @property
    def miss_rate(self):
        if self.alert_frames == 0:
            return 0.0
        return self.missed_frames / self.alert_frames

    def verdict(self):
        """Faz 0 basari kriterlerine gore gecti/kaldi."""
        checks = [
            ("carpisma yok", self.collisions == 0),
            ("yalanci alarm < 1/dk", self.false_alarms_per_minute < 1.0),
            ("kacirma < %5", self.miss_rate < 0.05),
        ]
        return checks, all(ok for _, ok in checks)

    def summary_lines(self):
        checks, passed = self.verdict()
        lines = [
            f"sure          {self.elapsed:6.1f} s",
            f"yol           {self.distance:6.1f} m",
            f"carpisma      {self.collisions:6d}",
            f"yalanci alarm {self.false_alarms:6d}  ({self.false_alarms_per_minute:.1f}/dk)",
            f"kacirma       {self.miss_rate * 100:6.1f} %",
        ]
        if self.min_clearance < float("inf"):
            lines.append(f"en yakin      {self.min_clearance:6.2f} m")
        if self.goal_reached:
            lines.append(f"HEDEFE VARILDI ({self.goal_time:.1f} s)")
        return lines, checks, passed
