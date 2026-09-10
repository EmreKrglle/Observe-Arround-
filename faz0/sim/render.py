"""
tkinter ile kusbakisi cizim.

Hicbir dis bagimlilik yok - tkinter Python'un standart kutuphanesinde.
"""

import math

import config
from haptics import patterns

PANEL_W = 300


def _lerp_color(c1, c2, t):
    """Iki hex rengi arasinda gecis."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    return "#%02x%02x%02x" % (
        int(r1 + (r2 - r1) * t),
        int(g1 + (g2 - g1) * t),
        int(b1 + (b2 - b1) * t),
    )


PATTERN_COLORS = {
    patterns.OBSTACLE: config.COL_WARN,
    patterns.EMERGENCY: config.COL_CRIT,
    patterns.GUIDANCE: "#4a9ed4",
    patterns.HEARTBEAT: "#4a7a5e",
    patterns.IDLE: config.COL_GRID,
}


class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas
        self.view_w = config.WINDOW_W - PANEL_W
        self.view_h = config.WINDOW_H
        self.ppm = config.PIXELS_PER_METER

    # -- koordinat donusumu ------------------------------------------------

    def to_screen(self, wx, wy, cam_x, cam_y):
        sx = self.view_w / 2.0 + (wx - cam_x) * self.ppm
        sy = self.view_h / 2.0 - (wy - cam_y) * self.ppm   # y ekranda ters
        return sx, sy

    # -- ana cizim ---------------------------------------------------------

    def draw(self, world, walker, readings, cmds, scorer, state):
        c = self.canvas
        c.delete("all")
        cam_x, cam_y = walker.x, walker.y

        self._grid(cam_x, cam_y)
        self._world(world, cam_x, cam_y)
        self._rays(walker, readings, cam_x, cam_y)
        self._body(walker, cam_x, cam_y)
        self._panel(cmds, scorer, state, walker)

    # -- parcalar ----------------------------------------------------------

    def _grid(self, cam_x, cam_y):
        c = self.canvas
        c.create_rectangle(0, 0, self.view_w, self.view_h,
                           fill=config.COL_BG, outline="")
        step = self.ppm
        ox = (self.view_w / 2.0 - cam_x * self.ppm) % step
        oy = (self.view_h / 2.0 + cam_y * self.ppm) % step
        x = ox
        while x < self.view_w:
            c.create_line(x, 0, x, self.view_h, fill=config.COL_GRID)
            x += step
        y = oy
        while y < self.view_h:
            c.create_line(0, y, self.view_w, y, fill=config.COL_GRID)
            y += step

    def _world(self, world, cam_x, cam_y):
        c = self.canvas
        for w in world.walls:
            x1, y1 = self.to_screen(w.ax, w.ay, cam_x, cam_y)
            x2, y2 = self.to_screen(w.bx, w.by, cam_x, cam_y)
            c.create_line(x1, y1, x2, y2, fill=config.COL_WALL, width=5)

        for p in world.pillars:
            x, y = self.to_screen(p.x, p.y, cam_x, cam_y)
            r = p.r * self.ppm
            c.create_oval(x - r, y - r, x + r, y + r,
                          fill=config.COL_PILLAR, outline="")

        for m in world.movers:
            x, y = self.to_screen(m.x, m.y, cam_x, cam_y)
            r = m.r * self.ppm
            c.create_oval(x - r, y - r, x + r, y + r,
                          fill=config.COL_MOVER, outline="")

        if world.goal:
            x, y = self.to_screen(world.goal[0], world.goal[1], cam_x, cam_y)
            r = world.goal_radius * self.ppm
            c.create_oval(x - r, y - r, x + r, y + r,
                          outline=config.COL_GOAL, width=3)
            c.create_text(x, y, text="HEDEF", fill=config.COL_GOAL,
                          font=("TkDefaultFont", 9, "bold"))

    def _rays(self, walker, readings, cam_x, cam_y):
        c = self.canvas
        ox, oy = self.to_screen(walker.x, walker.y, cam_x, cam_y)
        head = walker.sensed_heading
        for r in readings:
            ang = head + r.angle
            ex = walker.x + math.cos(ang) * r.distance
            ey = walker.y + math.sin(ang) * r.distance
            sx, sy = self.to_screen(ex, ey, cam_x, cam_y)
            if r.valid:
                c.create_line(ox, oy, sx, sy, fill=config.COL_HIT, width=2)
                c.create_oval(sx - 4, sy - 4, sx + 4, sy + 4,
                              fill=config.COL_HIT, outline="")
                c.create_text(sx + 12, sy - 10, text=f"{r.distance:.1f}",
                              fill=config.COL_HIT, anchor="w",
                              font=("TkDefaultFont", 8))
            else:
                c.create_line(ox, oy, sx, sy, fill=config.COL_RAY, dash=(3, 4))

    def _body(self, walker, cam_x, cam_y):
        c = self.canvas
        x, y = self.to_screen(walker.x, walker.y, cam_x, cam_y)
        r = config.BODY_RADIUS_M * self.ppm
        c.create_oval(x - r, y - r, x + r, y + r,
                      outline=config.COL_BODY, width=2)
        # Bakis yonu (salinim dahil - sensorlerin gercekte baktigi yon)
        h = walker.sensed_heading
        c.create_line(x, y,
                      x + math.cos(h) * r * 2.2,
                      y - math.sin(h) * r * 2.2,
                      fill=config.COL_BODY, width=3, arrow="last")

    # -- sag panel ---------------------------------------------------------

    def _panel(self, cmds, scorer, state, walker):
        c = self.canvas
        px = self.view_w
        c.create_rectangle(px, 0, config.WINDOW_W, config.WINDOW_H,
                           fill="#0c0e13", outline="")

        cx = px + PANEL_W / 2
        cy = 130
        self._motor_ring(cmds, cx, cy)

        y = 240
        c.create_text(px + 20, y, text="MOTORLAR", anchor="w",
                      fill=config.COL_DIM, font=("TkDefaultFont", 9, "bold"))
        y += 18
        for cmd in cmds:
            name = config.MOTOR_NAMES[cmd.index]
            bar_w = int(cmd.intensity * 120)
            col = PATTERN_COLORS.get(cmd.pattern, config.COL_GRID)
            c.create_text(px + 20, y, text=name, anchor="w",
                          fill=config.COL_TEXT, font=("TkDefaultFont", 9))
            c.create_rectangle(px + 95, y - 6, px + 215, y + 6,
                               fill=config.COL_GRID, outline="")
            if bar_w > 0:
                c.create_rectangle(px + 95, y - 6, px + 95 + bar_w, y + 6,
                                   fill=col, outline="")
            y += 20

        y += 10
        c.create_text(px + 20, y, text="OLCUMLER", anchor="w",
                      fill=config.COL_DIM, font=("TkDefaultFont", 9, "bold"))
        y += 18
        lines, checks, passed = scorer.summary_lines()
        for line in lines:
            c.create_text(px + 20, y, text=line, anchor="w",
                          fill=config.COL_TEXT, font=("TkFixedFont", 9))
            y += 16

        y += 8
        for label, ok in checks:
            c.create_text(px + 20, y, text=("[+] " if ok else "[-] ") + label,
                          anchor="w",
                          fill=(config.COL_GOAL if ok else config.COL_CRIT),
                          font=("TkFixedFont", 9))
            y += 16

        y += 12
        c.create_text(px + 20, y, text="ORTAM", anchor="w",
                      fill=config.COL_DIM, font=("TkDefaultFont", 9, "bold"))
        y += 18
        amb = state["ambient"]
        rng = config.sensor_max_range(amb)
        desc = ("ic mekan" if amb < 0.2 else
                "bulutlu" if amb < 0.55 else "direkt gunes")
        c.create_text(px + 20, y, anchor="w", fill=config.COL_TEXT,
                      font=("TkFixedFont", 9),
                      text=f"isik  {amb:.2f}  ({desc})")
        y += 16
        c.create_text(px + 20, y, anchor="w",
                      fill=(config.COL_CRIT if rng < 1.6 else config.COL_TEXT),
                      font=("TkFixedFont", 9),
                      text=f"menzil {rng:.2f} m")
        y += 16
        if rng < config.ALERT_DISTANCE_M:
            c.create_text(px + 20, y, anchor="w", fill=config.COL_CRIT,
                          font=("TkFixedFont", 8),
                          text="menzil < uyari mesafesi!")
            y += 14

        # Kontroller
        y = config.WINDOW_H - 150
        c.create_text(px + 20, y, text="KONTROLLER", anchor="w",
                      fill=config.COL_DIM, font=("TkDefaultFont", 9, "bold"))
        y += 16
        for line in ["yon tuslari  yuru / don",
                     "Q E          yana adim",
                     "[ ]          gunes isigi",
                     "G            hedef rehberi",
                     "R            bastan basla",
                     "bosluk       duraklat",
                     "ESC          cik"]:
            c.create_text(px + 20, y, text=line, anchor="w",
                          fill=config.COL_DIM, font=("TkFixedFont", 8))
            y += 14

        if state.get("paused"):
            c.create_text(self.view_w / 2, 40, text="DURAKLATILDI",
                          fill=config.COL_WARN,
                          font=("TkDefaultFont", 18, "bold"))

    def _motor_ring(self, cmds, cx, cy):
        """Govdeyi tepeden goren motor halkasi."""
        c = self.canvas
        R = 78
        c.create_oval(cx - R, cy - R, cx + R, cy + R,
                      outline=config.COL_GRID, width=2)
        c.create_text(cx, cy - R - 16, text="on", fill=config.COL_DIM,
                      font=("TkDefaultFont", 8))

        for cmd in cmds:
            ang = config.MOTOR_ANGLES_RAD[cmd.index]
            # Ekranda: 0 derece = yukari (on), pozitif = sola
            mx = cx - math.sin(ang) * R
            my = cy - math.cos(ang) * R
            base_r = 11
            r = base_r + cmd.intensity * 13
            col = PATTERN_COLORS.get(cmd.pattern, config.COL_GRID)
            if cmd.intensity <= 0.0:
                col = config.COL_GRID
            else:
                col = _lerp_color("#2a2f3a", col, 0.25 + cmd.intensity * 0.75)
            c.create_oval(mx - r, my - r, mx + r, my + r,
                          fill=col, outline="")
            c.create_text(mx, my, text=str(cmd.index),
                          fill=config.COL_BG if cmd.intensity > 0.4
                          else config.COL_DIM,
                          font=("TkDefaultFont", 8, "bold"))

        active = [c_ for c_ in cmds if c_.intensity > 0.0]
        label = active[0].pattern if active else patterns.IDLE
        c.create_text(cx, cy, text=label,
                      fill=PATTERN_COLORS.get(label, config.COL_DIM),
                      font=("TkDefaultFont", 10, "bold"))
