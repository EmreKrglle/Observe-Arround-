#!/usr/bin/env python3
"""
Sensor yerlesimlerini karsilastirir.

Her yerlesim, her senaryoda birden fazla seed ile ekransiz calistirilir.
Sensor gurultusu tek rastgelelik kaynagidir; seed sabit oldugu icin
tum yerlesimler birebir ayni kosullarda olculur.

Kullanim
--------
    python3 tools/compare_layouts.py
    python3 tools/compare_layouts.py --seconds 40 --seeds 10
    python3 tools/compare_layouts.py --ambient 1.0      # gunes altinda

Motorlar (config.MOTOR_ANGLES_DEG) degismez; sadece sensorler degisir.
Haptik katman sensor sayisini bilmez, sadece kerteriz listesi alir.
"""

import argparse
import contextlib
import io
import math
import os
import random
import sys

FAZ0_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, FAZ0_DIR)

import config                                        # noqa: E402
import run_sim                                       # noqa: E402
from haptics.backends import make_backend            # noqa: E402
from sim.sensors import ToFArray                     # noqa: E402

SCENARIOS = ["koridor", "dar_kapi", "acik_alan", "bos_oda"]

# Aci: 0 = tam on, pozitif = sol. Derece.
LAYOUTS = {
    "6 esit (mevcut)":   [0, 60, 120, 180, 240, 300],
    "8 esit":            [0, 45, 90, 135, 180, 225, 270, 315],
    "8 on-yogun 7+1":    [0, 30, 60, 90, 180, 270, 300, 330],
    "8 on-yogun 6+2":    [18, 54, 90, 150, 210, 270, 306, 342],
    "9 on-yogun 7+2":    [0, 30, 60, 90, 150, 210, 270, 300, 330],
    "10 on-yogun 7+3":   [0, 30, 60, 90, 135, 180, 225, 270, 300, 330],
    "12 esit":           list(range(0, 360, 30)),
    "16 esit":           [i * 22.5 for i in range(16)],
}


# ---------------------------------------------------------------------------
# Geometrik kapsama (simulasyonsuz)
# ---------------------------------------------------------------------------

def coverage(angles_deg, fov_deg, lo=-180.0, hi=180.0, step=0.1):
    """[lo, hi) araligindaki acilarin kacta kacini en az bir koni goruyor.

    Doner: (kapsama orani, en buyuk kor bosluk derece)
    """
    half = fov_deg / 2.0
    n = int(round((hi - lo) / step))
    covered = 0
    gap = best_gap = 0.0
    for k in range(n):
        a = lo + k * step
        seen = any(abs((a - s + 180.0) % 360.0 - 180.0) <= half for s in angles_deg)
        if seen:
            covered += 1
            gap = 0.0
        else:
            gap += step
            best_gap = max(best_gap, gap)
    return covered / n, best_gap


# ---------------------------------------------------------------------------
# Simulasyon
# ---------------------------------------------------------------------------

def run_once(scenario, angles_deg, seed, seconds, ambient):
    world = run_sim.load_scenario(scenario)
    sim = run_sim.Simulation(world, make_backend("sim"))
    sim.tof = ToFArray([math.radians(a) for a in angles_deg],
                       rng=random.Random(seed))
    if ambient is not None:
        sim.ambient = ambient
    with contextlib.redirect_stdout(io.StringIO()):
        run_sim.run_headless(sim, seconds)
    sim.backend.close()
    return sim.scorer


def evaluate(angles_deg, seeds, seconds, ambient):
    """Tum senaryolar x seed'ler. Doner: {senaryo: (carpisma ort, alarm/dk ort, kacirma ort)}"""
    out = {}
    for sc in SCENARIOS:
        col = fa = miss = 0.0
        for seed in range(seeds):
            s = run_once(sc, angles_deg, seed, seconds, ambient)
            col += s.collisions
            fa += s.false_alarms_per_minute
            miss += s.miss_rate
        out[sc] = (col / seeds, fa / seeds, miss / seeds)
    return out


def main():
    ap = argparse.ArgumentParser(description="Sensor yerlesimi karsilastirmasi")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ambient", type=float, default=None,
                    help="ortam isigi 0..1 (varsayilan config degeri)")
    args = ap.parse_args()

    fov = config.SENSOR_FOV_DEG
    print(f"FOV {fov:.0f} derece | {args.seeds} seed x {args.seconds:.0f} sn | "
          f"ortam isigi {args.ambient if args.ambient is not None else config.AMBIENT_LIGHT}\n")

    header = (f"{'yerlesim':<18} {'360 kaps.':>9} {'on kaps.':>9} {'on bosluk':>9}  "
              + "  ".join(f"{sc:>16}" for sc in SCENARIOS))
    print(header)
    print(" " * 50 + "  ".join(f"{'carp alarm kacir':>16}" for _ in SCENARIOS))
    print("-" * len(header))

    for name, angles in LAYOUTS.items():
        full, _ = coverage(angles, fov)
        front, front_gap = coverage(angles, fov, -90.0, 90.0)
        res = evaluate(angles, args.seeds, args.seconds, args.ambient)
        cells = "  ".join(f"{c:4.1f} {fa:4.1f} {m * 100:5.1f}%" for c, fa, m in res.values())
        print(f"{name:<18} {full * 100:8.0f}% {front * 100:8.0f}% {front_gap:8.1f}°  {cells}")

    print("\ncarp = seed basina ortalama carpisma, alarm = yalanci alarm/dk, "
          "kacir = kacirma orani")
    print("Hedef: carpisma 0, alarm < 1/dk, kacirma < %5")


if __name__ == "__main__":
    main()
