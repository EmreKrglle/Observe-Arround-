"""Geometri fonksiyonlarinin birim testleri.

Calistirmak icin:  python3 -m unittest discover faz0/tests
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sim.geometry import (          # noqa: E402
    angle_diff,
    normalize_angle,
    point_segment_distance,
    ray_circle,
    ray_segment,
    to_body_frame,
)


class TestAngles(unittest.TestCase):
    def test_normalize_wraps(self):
        self.assertAlmostEqual(normalize_angle(0.0), 0.0)
        self.assertAlmostEqual(normalize_angle(math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(-math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(3 * math.pi), math.pi)
        self.assertAlmostEqual(normalize_angle(math.radians(370)), math.radians(10))

    def test_diff_takes_short_way(self):
        # 350 derece ile 10 derece arasi 20 derecedir, 340 degil
        d = angle_diff(math.radians(10), math.radians(350))
        self.assertAlmostEqual(math.degrees(d), 20.0, places=6)


class TestRayCircle(unittest.TestCase):
    def test_head_on(self):
        # Orijinden +x yonune, merkezi (5,0) yaricapi 1 olan daire
        t = ray_circle(0, 0, 1, 0, 5, 0, 1)
        self.assertAlmostEqual(t, 4.0, places=6)

    def test_miss(self):
        self.assertIsNone(ray_circle(0, 0, 1, 0, 5, 10, 1))

    def test_behind_ray_is_ignored(self):
        # Daire isinin arkasinda
        self.assertIsNone(ray_circle(0, 0, 1, 0, -5, 0, 1))

    def test_tangent(self):
        # Tam tegetsel: yaricap kadar yandan gecen isin
        t = ray_circle(0, 0, 1, 0, 5, 1, 1)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 5.0, places=6)

    def test_origin_inside_circle(self):
        # Isin dairenin icinden basliyorsa cikis noktasini vermeli
        t = ray_circle(0, 0, 1, 0, 0, 0, 2)
        self.assertAlmostEqual(t, 2.0, places=6)


class TestRaySegment(unittest.TestCase):
    def test_perpendicular_wall(self):
        # x=3'te duvar, y=-1..1 arasi; +x yonune giden isin
        t = ray_segment(0, 0, 1, 0, 3, -1, 3, 1)
        self.assertAlmostEqual(t, 3.0, places=6)

    def test_misses_short_wall(self):
        # Duvar y=5..6 arasinda, isin y=0 hizasinda gidiyor
        self.assertIsNone(ray_segment(0, 0, 1, 0, 3, 5, 3, 6))

    def test_parallel(self):
        self.assertIsNone(ray_segment(0, 0, 1, 0, 0, 1, 5, 1))

    def test_behind(self):
        self.assertIsNone(ray_segment(0, 0, 1, 0, -3, -1, -3, 1))

    def test_diagonal(self):
        # 45 derece giden isin, x=2 duvarina 2*sqrt(2) mesafede carpar
        s = math.sqrt(0.5)
        t = ray_segment(0, 0, s, s, 2, 0, 2, 5)
        self.assertAlmostEqual(t, 2 * math.sqrt(2), places=6)


class TestPointSegment(unittest.TestCase):
    def test_perpendicular_foot(self):
        d = point_segment_distance(0, 3, -5, 0, 5, 0)
        self.assertAlmostEqual(d, 3.0, places=6)

    def test_past_the_end_uses_endpoint(self):
        # Parcanin disinda kalan nokta en yakin uc noktaya olculur
        d = point_segment_distance(10, 0, -5, 0, 5, 0)
        self.assertAlmostEqual(d, 5.0, places=6)

    def test_degenerate_segment(self):
        d = point_segment_distance(3, 4, 0, 0, 0, 0)
        self.assertAlmostEqual(d, 5.0, places=6)


class TestBodyFrame(unittest.TestCase):
    def test_straight_ahead(self):
        # Kuzeye bakiyoruz (90 derece), hedef tam kuzeyde
        dist, bearing = to_body_frame(0, 5, 0, 0, math.radians(90))
        self.assertAlmostEqual(dist, 5.0, places=6)
        self.assertAlmostEqual(math.degrees(bearing), 0.0, places=6)

    def test_object_on_the_left(self):
        # Doguya bakiyoruz (0 derece), hedef kuzeyde -> SOLUMUZDA
        dist, bearing = to_body_frame(0, 5, 0, 0, 0.0)
        self.assertAlmostEqual(dist, 5.0, places=6)
        self.assertAlmostEqual(math.degrees(bearing), 90.0, places=6)

    def test_object_on_the_right(self):
        # Doguya bakiyoruz, hedef guneyde -> SAGIMIZDA (negatif)
        dist, bearing = to_body_frame(0, -5, 0, 0, 0.0)
        self.assertAlmostEqual(math.degrees(bearing), -90.0, places=6)

    def test_behind(self):
        dist, bearing = to_body_frame(-5, 0, 0, 0, 0.0)
        self.assertAlmostEqual(abs(math.degrees(bearing)), 180.0, places=6)


if __name__ == "__main__":
    unittest.main()
