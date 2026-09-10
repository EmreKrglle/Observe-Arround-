#!/usr/bin/env python3
"""
Faz 0 - 2B navigasyon simulatoru

Kullanim
--------
    python3 run_sim.py                              # koridor senaryosu
    python3 run_sim.py --scenario dar_kapi
    python3 run_sim.py --backend console            # ekransiz, terminale bas
    python3 run_sim.py --backend hardware           # GERCEK motorlari sur
    python3 run_sim.py --headless 30                # 30 sn otomatik, puan bas
    python3 run_sim.py --test-motors                # motorlari tek tek dene

Donanim-dongude test (projenin en degerli kullanimi):
    Sanal dunyayi ekranda tutup GERCEK bandi surersiniz. Gozu bagli bir
    gonullu bandi takar, siz klavyeden yurutursunuz. Boylece simule
    edemeyeceginiz tek sey - insan algisi - gercek kalir.

        python3 run_sim.py --backend hardware --scenario koridor
"""

import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config                                       # noqa: E402
from haptics.backends import make_backend            # noqa: E402
from haptics.mapper import map_obstacles             # noqa: E402
from sim.geometry import to_body_frame               # noqa: E402
from sim.scoring import Scorer                       # noqa: E402
from sim.sensors import ToFArray, readings_to_obstacles   # noqa: E402
from sim.walker import Walker                        # noqa: E402
from sim.world import World                          # noqa: E402

SCENARIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios")


def load_scenario(name):
    path = name if os.path.isfile(name) else os.path.join(SCENARIO_DIR, name + ".json")
    if not os.path.isfile(path):
        available = sorted(f[:-5] for f in os.listdir(SCENARIO_DIR)
                           if f.endswith(".json"))
        raise SystemExit(f"Senaryo bulunamadi: {name}\nMevcut: {', '.join(available)}")
    return World.load(path)


class Simulation:
    """Ekrandan bagimsiz simulasyon cekirdegi.

    Hem grafik arayuz hem headless mod bunu kullanir - boylece
    ikisi ayni sonucu uretir.
    """

    def __init__(self, world, backend, use_goal=True):
        self.world = world
        self.backend = backend
        self.tof = ToFArray()
        self.walker = Walker(*world.start, heading_deg=world.start_heading_deg)
        self.scorer = Scorer()
        self.t = 0.0
        self.ambient = config.AMBIENT_LIGHT
        self.use_goal = use_goal
        self.readings = []
        self.cmds = []

    def reset(self):
        self.walker.reset(*self.world.start, heading_deg=self.world.start_heading_deg)
        self.scorer.reset()
        self.t = 0.0

    def goal_bearing(self):
        if not self.use_goal or self.world.goal is None:
            return None
        if self.world.reached_goal(self.walker.x, self.walker.y):
            return None
        _, bearing = to_body_frame(self.world.goal[0], self.world.goal[1],
                                   self.walker.x, self.walker.y,
                                   self.walker.heading)
        return bearing

    def step(self, dt, forward=0.0, turn=0.0, strafe=0.0):
        self.t += dt
        self.world.update(dt)
        self.walker.update(dt, forward, turn, strafe)

        # Sensorler govdeyle birlikte salinir - sistem bunu bilmez
        self.readings = self.tof.sense(
            self.walker.x, self.walker.y, self.walker.sensed_heading,
            self.world, self.ambient,
        )
        obstacles = readings_to_obstacles(self.readings)
        self.cmds = map_obstacles(obstacles, self.t, self.goal_bearing())
        self.backend.send(self.cmds)
        self.scorer.update(dt, self.walker, self.world, obstacles)
        return self.cmds


# ---------------------------------------------------------------------------
# Grafik arayuz
# ---------------------------------------------------------------------------

class App:
    def __init__(self, sim):
        import tkinter as tk
        from sim.render import Renderer

        self.tk = tk
        self.sim = sim
        self.root = tk.Tk()
        self.root.title(f"Faz 0 simulatoru - {sim.world.name}")
        self.root.configure(bg=config.COL_BG)
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(self.root, width=config.WINDOW_W,
                                height=config.WINDOW_H, highlightthickness=0,
                                bg=config.COL_BG)
        self.canvas.pack()
        self.renderer = Renderer(self.canvas)

        self.keys = set()
        self.paused = False
        self.root.bind("<KeyPress>", self._key_down)
        self.root.bind("<KeyRelease>", self._key_up)
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        self._last = time.monotonic()
        self._dt_target = 1.0 / config.FPS

    def _key_down(self, e):
        k = e.keysym.lower()
        self.keys.add(k)
        if k == "escape":
            self.quit()
        elif k == "space":
            self.paused = not self.paused
        elif k == "r":
            self.sim.reset()
        elif k == "g":
            self.sim.use_goal = not self.sim.use_goal
        elif k == "bracketleft":
            self.sim.ambient = max(0.0, self.sim.ambient - 0.1)
        elif k == "bracketright":
            self.sim.ambient = min(1.0, self.sim.ambient + 0.1)

    def _key_up(self, e):
        self.keys.discard(e.keysym.lower())

    def _input(self):
        f = t = s = 0.0
        if "up" in self.keys or "w" in self.keys:
            f += 1.0
        if "down" in self.keys or "s" in self.keys:
            f -= 1.0
        if "left" in self.keys or "a" in self.keys:
            t += 1.0        # sola donus = pozitif aci
        if "right" in self.keys or "d" in self.keys:
            t -= 1.0
        if "q" in self.keys:
            s += 1.0
        if "e" in self.keys:
            s -= 1.0
        return f, t, s

    def tick(self):
        now = time.monotonic()
        dt = min(now - self._last, 0.1)      # buyuk sicramalari kirp
        self._last = now

        if not self.paused:
            self.sim.step(dt, *self._input())

        self.renderer.draw(self.sim.world, self.sim.walker, self.sim.readings,
                           self.sim.cmds, self.sim.scorer,
                           {"ambient": self.sim.ambient, "paused": self.paused})
        self.root.after(int(self._dt_target * 1000), self.tick)

    def run(self):
        self.tick()
        try:
            self.root.mainloop()
        finally:
            self.sim.backend.close()

    def quit(self):
        self.sim.backend.close()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Ekransiz mod
# ---------------------------------------------------------------------------

def run_headless(sim, seconds):
    """Otomatik yuruyus: duz git, onde engel varsa don.

    Amaci gercekci bir gezgin olmak degil - butun boru hattini
    (sensor -> esleme -> motor -> puan) ekransiz calistirmak.
    Surum degisikliklerinden sonra bozulma olup olmadigini boyle test edin.
    """
    dt = 1.0 / config.FPS
    steps = int(seconds / dt)
    turn_bias = 1.0

    for _ in range(steps):
        front = [d for a, d in readings_to_obstacles(sim.readings)
                 if abs(a) < math.radians(40)]
        blocked = front and min(front) < 1.2
        if blocked:
            sim.step(dt, forward=0.15, turn=turn_bias)
        else:
            sim.step(dt, forward=1.0, turn=0.0)
            turn_bias = 1.0 if (sim.t % 7.0) < 3.5 else -1.0

    lines, checks, passed = sim.scorer.summary_lines()
    print(f"\n=== {sim.world.name} ===")
    for line in lines:
        print("  " + line)
    print()
    for label, ok in checks:
        print(f"  [{'+' if ok else '-'}] {label}")
    print(f"\n  SONUC: {'GECTI' if passed else 'KALDI'}\n")
    return passed


def test_motors(backend):
    """Motorlari tek tek calistir - kablolamayi dogrulamak icin."""
    from haptics.mapper import MotorCommand
    from haptics import patterns
    n = len(config.MOTOR_ANGLES_DEG)
    print("Motorlar sirayla calistirilacak. Her birini elle kontrol edin.\n")
    for i in range(n):
        cmds = [MotorCommand(k, 1.0 if k == i else 0.0,
                             patterns.OBSTACLE if k == i else patterns.IDLE)
                for k in range(n)]
        print(f"  motor {i} -> {config.MOTOR_NAMES[i]}")
        backend.send(cmds)
        time.sleep(1.0)
    backend.stop()
    print("\nBitti. Yanlis yerde titreyen motor varsa kablolari degistirin\n"
          "ya da config.py'deki MOTOR_ANGLES_DEG sirasini duzeltin.")


def main():
    ap = argparse.ArgumentParser(description="Faz 0 navigasyon simulatoru")
    ap.add_argument("--scenario", default="koridor",
                    help="senaryo adi ya da .json yolu")
    ap.add_argument("--backend", default="sim",
                    choices=["sim", "console", "hardware"],
                    help="motor cikisi: sim (ekran), console (terminal), "
                         "hardware (gercek motorlar)")
    ap.add_argument("--headless", type=float, metavar="SANIYE",
                    help="grafik arayuz olmadan otomatik calistir")
    ap.add_argument("--test-motors", action="store_true",
                    help="motorlari tek tek dene ve cik")
    ap.add_argument("--ambient", type=float, default=None,
                    help="ortam isigi 0..1 (0=ic mekan, 1=direkt gunes)")
    ap.add_argument("--no-goal", action="store_true",
                    help="hedef rehberligini kapat, sadece engel kacinma")
    args = ap.parse_args()

    backend = make_backend(args.backend)

    if args.test_motors:
        test_motors(backend)
        backend.close()
        return 0

    world = load_scenario(args.scenario)
    sim = Simulation(world, backend, use_goal=not args.no_goal)
    if args.ambient is not None:
        sim.ambient = max(0.0, min(1.0, args.ambient))

    if args.headless is not None:
        try:
            ok = run_headless(sim, args.headless)
        finally:
            backend.close()
        return 0 if ok else 1

    try:
        App(sim).run()
    except Exception as exc:          # tkinter yoksa ya da ekran yoksa
        backend.close()
        print(f"Grafik arayuz baslatilamadi: {exc}\n"
              f"Ekransiz denemek icin:  python3 run_sim.py --headless 30")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
