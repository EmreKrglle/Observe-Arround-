"""
Sanal dunya: duvarlar, direkler, hareketli engeller ve hedef.

Senaryolar JSON dosyalarindan yuklenir (faz0/scenarios/ klasoru).
"""

import json
import math

from .geometry import point_segment_distance


class Wall:
    """Iki nokta arasinda duz duvar."""

    kind = "wall"

    def __init__(self, ax, ay, bx, by):
        self.ax, self.ay, self.bx, self.by = ax, ay, bx, by

    def distance_to(self, px, py):
        return point_segment_distance(px, py, self.ax, self.ay, self.bx, self.by)


class Pillar:
    """Direk, agac, cop kutusu - dairesel engel."""

    kind = "pillar"

    def __init__(self, x, y, r):
        self.x, self.y, self.r = x, y, r

    def distance_to(self, px, py):
        return math.hypot(px - self.x, py - self.y) - self.r


class Mover:
    """Hareketli engel - yaya, bisikletli.

    Iki nokta arasinda gidip gelir. Radar'in en iyi gordugu,
    ToF'un en zor takip ettigi nesne tipi.
    """

    kind = "mover"

    def __init__(self, x0, y0, x1, y1, r, speed):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.r = r
        self.speed = speed
        self.x, self.y = x0, y0
        self._t = 0.0          # 0..1 arasi yol orani
        self._dir = 1.0

    def update(self, dt):
        leg = math.hypot(self.x1 - self.x0, self.y1 - self.y0)
        if leg < 1e-9:
            return
        self._t += self._dir * (self.speed * dt) / leg
        if self._t >= 1.0:
            self._t, self._dir = 1.0, -1.0
        elif self._t <= 0.0:
            self._t, self._dir = 0.0, 1.0
        self.x = self.x0 + (self.x1 - self.x0) * self._t
        self.y = self.y0 + (self.y1 - self.y0) * self._t

    def distance_to(self, px, py):
        return math.hypot(px - self.x, py - self.y) - self.r


class World:
    """Bir senaryonun tamami."""

    def __init__(self, name="isimsiz"):
        self.name = name
        self.walls = []
        self.pillars = []
        self.movers = []
        self.start = (1.0, 1.0)
        self.start_heading_deg = 0.0
        self.goal = None
        self.goal_radius = 0.6
        self.description = ""

    # -- yukleme -----------------------------------------------------------

    @classmethod
    def from_dict(cls, data):
        w = cls(data.get("name", "isimsiz"))
        w.description = data.get("description", "")
        for a in data.get("walls", []):
            w.walls.append(Wall(a[0], a[1], a[2], a[3]))
        for p in data.get("pillars", []):
            w.pillars.append(Pillar(p[0], p[1], p[2]))
        for m in data.get("movers", []):
            w.movers.append(Mover(m[0], m[1], m[2], m[3], m[4], m[5]))
        w.start = tuple(data.get("start", [1.0, 1.0]))
        w.start_heading_deg = data.get("start_heading_deg", 0.0)
        if "goal" in data:
            w.goal = tuple(data["goal"])
            w.goal_radius = data.get("goal_radius", 0.6)
        return w

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    # -- sorgular ----------------------------------------------------------

    @property
    def obstacles(self):
        """Tum engeller tek listede."""
        return self.walls + self.pillars + self.movers

    def update(self, dt):
        for m in self.movers:
            m.update(dt)

    def min_clearance(self, px, py):
        """En yakin engele olan mesafe (metre).

        Engel yoksa cok buyuk bir sayi doner.
        """
        best = float("inf")
        for o in self.obstacles:
            d = o.distance_to(px, py)
            if d < best:
                best = d
        return best

    def is_colliding(self, px, py, body_radius):
        return self.min_clearance(px, py) <= body_radius

    def reached_goal(self, px, py):
        if self.goal is None:
            return False
        return math.hypot(px - self.goal[0], py - self.goal[1]) <= self.goal_radius

    def bounds(self):
        """Dunyanin sinirlari (minx, miny, maxx, maxy) - kamera icin."""
        xs, ys = [], []
        for w in self.walls:
            xs += [w.ax, w.bx]
            ys += [w.ay, w.by]
        for p in self.pillars:
            xs += [p.x - p.r, p.x + p.r]
            ys += [p.y - p.r, p.y + p.r]
        for m in self.movers:
            xs += [m.x0, m.x1]
            ys += [m.y0, m.y1]
        if not xs:
            return (0.0, 0.0, 10.0, 10.0)
        return (min(xs), min(ys), max(xs), max(ys))
