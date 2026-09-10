"""Cizim katmani testleri.

Gercek bir tkinter penceresi acmadan calisir: sahte bir Canvas
nesnesi kullanip cizim cagrilarini kaydeder. Amac guzel gorunup
gorunmedigini olcmek degil - kodun HATA VERMEDIGINI dogrulamak.

Bir yazim hatasi ya da eksik degisken varsa bu test yakalar.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config                                        # noqa: E402
from haptics.backends import SimBackend               # noqa: E402
from sim.render import Renderer, _lerp_color, PANEL_W  # noqa: E402
from sim.scoring import Scorer                        # noqa: E402
from sim.walker import Walker                         # noqa: E402
from sim.world import World                           # noqa: E402


class FakeCanvas:
    """tkinter.Canvas yerine gecen kayit tutucu."""

    def __init__(self):
        self.calls = []

    def _record(self, name):
        def fn(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return len(self.calls)
        return fn

    def __getattr__(self, name):
        if name.startswith("create_") or name == "delete":
            return self._record(name)
        raise AttributeError(name)


SCENARIO = {
    "name": "test",
    "start": [1.0, 1.0],
    "walls": [[0, 0, 5, 0], [0, 0, 0, 5]],
    "pillars": [[2.0, 2.0, 0.3]],
    "movers": [[3.0, 1.0, 3.0, 4.0, 0.3, 1.0]],
    "goal": [4.0, 4.0],
}


class TestColor(unittest.TestCase):
    def test_endpoints(self):
        self.assertEqual(_lerp_color("#000000", "#ffffff", 0.0), "#000000")
        self.assertEqual(_lerp_color("#000000", "#ffffff", 1.0), "#ffffff")

    def test_midpoint(self):
        self.assertEqual(_lerp_color("#000000", "#ffffff", 0.5), "#7f7f7f")

    def test_clamped(self):
        self.assertEqual(_lerp_color("#000000", "#ffffff", -5.0), "#000000")
        self.assertEqual(_lerp_color("#000000", "#ffffff", 9.0), "#ffffff")

    def test_output_is_valid_hex(self):
        for t in [0.0, 0.13, 0.5, 0.77, 1.0]:
            col = _lerp_color(config.COL_BG, config.COL_CRIT, t)
            self.assertEqual(len(col), 7)
            self.assertTrue(col.startswith("#"))
            int(col[1:], 16)          # gecerli hex mi


class TestScreenTransform(unittest.TestCase):
    def setUp(self):
        self.r = Renderer(FakeCanvas())

    def test_camera_center_maps_to_view_center(self):
        sx, sy = self.r.to_screen(5.0, 5.0, 5.0, 5.0)
        self.assertAlmostEqual(sx, self.r.view_w / 2.0)
        self.assertAlmostEqual(sy, self.r.view_h / 2.0)

    def test_y_axis_is_flipped(self):
        """Dunyada YUKARI, ekranda YUKARI gorunmeli (kucuk y)."""
        _, sy_up = self.r.to_screen(0.0, 1.0, 0.0, 0.0)
        _, sy_down = self.r.to_screen(0.0, -1.0, 0.0, 0.0)
        self.assertLess(sy_up, sy_down)

    def test_x_axis_is_not_flipped(self):
        sx_right, _ = self.r.to_screen(1.0, 0.0, 0.0, 0.0)
        sx_left, _ = self.r.to_screen(-1.0, 0.0, 0.0, 0.0)
        self.assertGreater(sx_right, sx_left)

    def test_scale_matches_pixels_per_meter(self):
        x0, _ = self.r.to_screen(0.0, 0.0, 0.0, 0.0)
        x1, _ = self.r.to_screen(1.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(x1 - x0, config.PIXELS_PER_METER)

    def test_panel_does_not_overlap_view(self):
        self.assertEqual(self.r.view_w + PANEL_W, config.WINDOW_W)


class TestDrawSmoke(unittest.TestCase):
    """draw() hicbir durumda patlamamali."""

    def _run_draw(self, **overrides):
        from run_sim import Simulation
        world = World.from_dict(SCENARIO)
        sim = Simulation(world, SimBackend())
        sim.step(1.0 / 30.0, forward=1.0)
        canvas = FakeCanvas()
        r = Renderer(canvas)
        state = {"ambient": 0.0, "paused": False}
        state.update(overrides)
        r.draw(world, sim.walker, sim.readings, sim.cmds, sim.scorer, state)
        return canvas

    def test_draws_something(self):
        canvas = self._run_draw()
        self.assertGreater(len(canvas.calls), 20)

    def test_first_call_clears(self):
        canvas = self._run_draw()
        self.assertEqual(canvas.calls[0][0], "delete")

    @staticmethod
    def _all_text(canvas):
        """Cizilen tum metinleri toplar.

        create_text metni 'text=' anahtar kelimesiyle alir, konumu
        pozisyonel olarak - ikisine de bakmak gerekir.
        """
        out = []
        for name, args, kwargs in canvas.calls:
            if name != "create_text":
                continue
            if "text" in kwargs:
                out.append(str(kwargs["text"]))
            out.extend(str(a) for a in args)
        return " ".join(out)

    def test_paused_overlay(self):
        canvas = self._run_draw(paused=True)
        self.assertIn("DURAKLA", self._all_text(canvas))

    def test_not_paused_has_no_overlay(self):
        canvas = self._run_draw(paused=False)
        self.assertNotIn("DURAKLA", self._all_text(canvas))

    def test_bright_sun_warns(self):
        canvas = self._run_draw(ambient=1.0)
        self.assertIn("gunes", self._all_text(canvas))

    def test_motor_names_are_shown(self):
        canvas = self._run_draw()
        text = self._all_text(canvas)
        for name in config.MOTOR_NAMES:
            self.assertIn(name, text)

    def test_empty_world_does_not_crash(self):
        world = World.from_dict({"name": "bos", "start": [0, 0]})
        walker = Walker(0, 0, 0)
        canvas = FakeCanvas()
        Renderer(canvas).draw(world, walker, [], [], Scorer(),
                              {"ambient": 0.0, "paused": False})
        self.assertGreater(len(canvas.calls), 0)


if __name__ == "__main__":
    unittest.main()
