"""Puanlayici testleri.

Ozellikle: engel kalibi nabiz seklinde oldugu icin motorun kapali
oldugu anlar kacirma ya da yeni yalanci alarm sayilmamali.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                        # noqa: E402
from sim.scoring import Scorer                       # noqa: E402

DT = 1.0 / 30


class _Walker:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = x, y


class _World:
    """En yakin engele sabit mesafe veren sahte dunya."""

    def __init__(self, clearance):
        self.clearance = clearance

    def min_clearance(self, px, py):
        return self.clearance

    def reached_goal(self, px, py):
        return False


def _run(scorer, clearance, obstacles, frames):
    walker, world = _Walker(), _World(clearance)
    for _ in range(frames):
        scorer.update(DT, walker, world, obstacles)


class TestMissRate(unittest.TestCase):
    def test_seen_obstacle_is_never_missed(self):
        # Engel uyari mesafesinde ve sensor onu goruyor: nabzin kapali
        # anlari dahil hicbir kare kacirma sayilmamali.
        s = Scorer()
        _run(s, 1.0, [(0.0, 1.0)], frames=90)
        self.assertEqual(s.miss_rate, 0.0)

    def test_unseen_obstacle_is_missed(self):
        s = Scorer()
        _run(s, 1.0, [], frames=90)
        self.assertEqual(s.miss_rate, 1.0)

    def test_obstacle_beyond_alert_distance_does_not_count(self):
        s = Scorer()
        far = config.ALERT_DISTANCE_M + 0.5
        _run(s, far, [], frames=90)
        self.assertEqual(s.alert_frames, 0)


class TestFalseAlarms(unittest.TestCase):
    def test_continuous_false_alert_counts_once(self):
        # Ortada bir sey yokken 3 saniye boyunca uyari: tek yalanci alarm.
        # Her nabiz ayri sayilsaydi 2-9 Hz'de onlarca olurdu.
        s = Scorer()
        far = config.ALERT_DISTANCE_M + 1.0
        _run(s, far, [(0.0, 1.5)], frames=90)
        self.assertEqual(s.false_alarms, 1)

    def test_two_separate_false_alerts_count_twice(self):
        s = Scorer()
        far = config.ALERT_DISTANCE_M + 1.0
        _run(s, far, [(0.0, 1.5)], frames=10)
        _run(s, far, [], frames=10)
        _run(s, far, [(0.0, 1.5)], frames=10)
        self.assertEqual(s.false_alarms, 2)

    def test_real_obstacle_is_not_false_alarm(self):
        s = Scorer()
        _run(s, 1.5, [(0.0, 1.5)], frames=90)
        self.assertEqual(s.false_alarms, 0)


if __name__ == "__main__":
    unittest.main()
