"""Motor esleme mantiginin birim testleri.

Bu testler simulatorun en degerli parcasi: yon esleme hatasini
donanim almadan, gonullu yormadan burada yakalarsiniz.
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config                                    # noqa: E402
from haptics import patterns                      # noqa: E402
from haptics.mapper import (                      # noqa: E402
    _spread_weights,
    map_obstacles,
    proximity_of,
)

D = math.radians          # kisaltma: dereceyi radyana cevir
MOTORS = config.MOTOR_ANGLES_RAD

# t=0'da nabizlarin hepsi ACIK durumda (phase 0 < duty), test icin uygun.
T_ON = 0.0


class TestSpreadWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        for deg in range(0, 360, 7):
            w = _spread_weights(D(deg), MOTORS)
            self.assertAlmostEqual(sum(w), 1.0, places=9,
                                   msg=f"{deg} derecede toplam 1 degil")

    def test_exact_motor_gets_everything(self):
        # Tam 60 derece = 1 numarali motorun uzerinde
        w = _spread_weights(D(60), MOTORS)
        self.assertAlmostEqual(w[1], 1.0, places=9)
        self.assertAlmostEqual(sum(w) - w[1], 0.0, places=9)

    def test_halfway_splits_evenly(self):
        # 90 derece = 60 ile 120 arasinin tam ortasi
        w = _spread_weights(D(90), MOTORS)
        self.assertAlmostEqual(w[1], 0.5, places=9)
        self.assertAlmostEqual(w[2], 0.5, places=9)

    def test_wraps_around_zero(self):
        # 350 derece, 0 ile 300 numarali motorlar arasinda kalmali
        w = _spread_weights(D(350), MOTORS)
        self.assertGreater(w[0], 0.0)   # on
        self.assertGreater(w[5], 0.0)   # on-sag
        self.assertAlmostEqual(w[2], 0.0, places=9)
        self.assertAlmostEqual(w[3], 0.0, places=9)

    def test_only_two_motors_active(self):
        for deg in range(0, 360, 11):
            w = _spread_weights(D(deg), MOTORS)
            active = [x for x in w if x > 1e-9]
            self.assertLessEqual(len(active), 2,
                                 msg=f"{deg} derecede ikiden fazla motor aktif")


class TestProximity(unittest.TestCase):
    def test_endpoints(self):
        self.assertAlmostEqual(proximity_of(config.ALERT_DISTANCE_M), 0.0)
        self.assertAlmostEqual(proximity_of(config.CRITICAL_DISTANCE_M), 1.0)

    def test_clamped(self):
        self.assertEqual(proximity_of(99.0), 0.0)
        self.assertEqual(proximity_of(0.0), 1.0)

    def test_monotonic(self):
        # Yaklastikca yakinlik artmali
        prev = -1.0
        d = config.ALERT_DISTANCE_M
        while d >= config.CRITICAL_DISTANCE_M:
            p = proximity_of(d)
            self.assertGreaterEqual(p, prev)
            prev = p
            d -= 0.1


class TestMapObstacles(unittest.TestCase):
    def test_obstacle_ahead_fires_front_motor(self):
        cmds = map_obstacles([(D(0), 1.5)], T_ON)
        self.assertGreater(cmds[0].intensity, 0.0)
        self.assertEqual(cmds[0].pattern, patterns.OBSTACLE)
        # Arka motorlar sessiz olmali
        self.assertEqual(cmds[3].intensity, 0.0)

    def test_obstacle_left_fires_left_motors(self):
        cmds = map_obstacles([(D(90), 1.5)], T_ON)
        self.assertGreater(cmds[1].intensity, 0.0)   # on-sol
        self.assertGreater(cmds[2].intensity, 0.0)   # arka-sol
        self.assertEqual(cmds[5].intensity, 0.0)     # on-sag sessiz

    def test_obstacle_right_fires_right_motor(self):
        cmds = map_obstacles([(D(-90), 1.5)], T_ON)
        self.assertGreater(cmds[4].intensity, 0.0)   # arka-sag
        self.assertGreater(cmds[5].intensity, 0.0)   # on-sag
        self.assertEqual(cmds[1].intensity, 0.0)     # on-sol sessiz

    def test_left_and_right_are_not_mirrored_wrongly(self):
        """Yon isareti hatasi en sik yapilan ve en tehlikeli hatadir."""
        left = map_obstacles([(D(60), 1.5)], T_ON)
        right = map_obstacles([(D(-60), 1.5)], T_ON)
        # Sol engelde 1 numarali (on-sol) motor calismali
        self.assertGreater(left[1].intensity, 0.0)
        self.assertEqual(left[5].intensity, 0.0)
        # Sag engelde 5 numarali (on-sag) motor calismali
        self.assertGreater(right[5].intensity, 0.0)
        self.assertEqual(right[1].intensity, 0.0)

    def test_far_obstacle_ignored(self):
        far = config.ALERT_DISTANCE_M + 1.0
        cmds = map_obstacles([(D(0), far)], T_ON)
        self.assertTrue(all(c.intensity == 0.0 for c in cmds))

    def test_critical_triggers_all_motors(self):
        near = config.CRITICAL_DISTANCE_M - 0.1
        cmds = map_obstacles([(D(0), near)], T_ON)
        self.assertTrue(all(c.pattern == patterns.EMERGENCY for c in cmds))
        self.assertTrue(all(c.intensity > 0.5 for c in cmds))

    def test_closer_is_stronger(self):
        weak = map_obstacles([(D(0), 1.9)], T_ON)[0].intensity
        strong = map_obstacles([(D(0), 0.8)], T_ON)[0].intensity
        self.assertGreater(strong, weak)

    def test_guidance_does_not_share_motor_with_obstacle(self):
        # Hem engel hem hedef tam onde: engel kazanmali
        cmds = map_obstacles([(D(0), 1.2)], T_ON, goal_bearing=D(0))
        self.assertEqual(cmds[0].pattern, patterns.OBSTACLE)

    def test_guidance_alone_uses_guidance_pattern(self):
        cmds = map_obstacles([], T_ON, goal_bearing=D(120))
        self.assertEqual(cmds[2].pattern, patterns.GUIDANCE)
        self.assertGreater(cmds[2].intensity, 0.0)

    def test_nothing_to_say_is_quiet(self):
        # t=1.0'da kalp atisi penceresi disindayiz
        cmds = map_obstacles([], 1.0)
        self.assertTrue(all(c.intensity == 0.0 for c in cmds))

    def test_intensity_never_exceeds_one(self):
        many = [(D(a), 0.7) for a in range(0, 360, 15)]
        for t in [0.0, 0.03, 0.11, 0.27]:
            for c in map_obstacles(many, t):
                self.assertLessEqual(c.intensity, 1.0)
                self.assertGreaterEqual(c.intensity, 0.0)


if __name__ == "__main__":
    unittest.main()
