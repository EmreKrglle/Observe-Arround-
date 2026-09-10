"""
Engel listesini motor komutlarina cevirir.

Bu dosya simulatorde de gercek donanimda da AYNEN calisir - motorlarin
gercek mi sanal mi oldugunu bilmez. Simulasyonda dogruladiginiz mantik,
banda taktiginizda birebir ayni mantiktir.

Girdi : [(kerteriz_radyan, mesafe_metre), ...]
Cikti : [MotorCommand, ...]  (motor sayisi kadar)
"""

import math

import config
from sim.geometry import angle_diff
from . import patterns


class MotorCommand:
    """Tek bir motora gidecek komut."""

    __slots__ = ("index", "intensity", "pattern")

    def __init__(self, index, intensity=0.0, pattern=patterns.IDLE):
        self.index = index
        self.intensity = intensity      # 0.0 .. 1.0
        self.pattern = pattern          # hangi kalip - kayit ve ekran icin

    def __repr__(self):
        return f"<M{self.index} {self.intensity:.2f} {self.pattern}>"


def _spread_weights(bearing, motor_angles):
    """Bir kerterizi motorlara dagitir.

    Hayalet dokunus (phantom sensation): iki motor arasina dusen bir yon,
    ikisini kismi siddetle titretirsek insan tarafindan ARADAKI tek bir
    nokta olarak algilanir. Bu sayede 6 motorla 60 dereceden daha ince
    yon cozunurlugu elde edilir.

    Doner: motor sayisi uzunlugunda agirlik listesi, toplami 1.0
    """
    n = len(motor_angles)
    if n == 0:
        return []

    if not config.USE_PHANTOM_INTERPOLATION:
        # Sadece en yakin motor
        best_i = min(range(n), key=lambda k: abs(angle_diff(bearing, motor_angles[k])))
        w = [0.0] * n
        w[best_i] = 1.0
        return w

    spacing = 2.0 * math.pi / n
    weights = []
    for a in motor_angles:
        d = abs(angle_diff(bearing, a))
        # Bir motor araligi boyunca dogrusal sonumleme.
        # Tam motor uzerinde 1.0, bir motor otede 0.0.
        weights.append(max(0.0, 1.0 - d / spacing))

    total = sum(weights)
    if total <= 1e-9:
        return [0.0] * n
    return [w / total for w in weights]


def proximity_of(distance):
    """Mesafeyi 0..1 yakinlik degerine cevirir.

    uyari mesafesinde 0.0, kritik mesafede 1.0
    """
    far = config.ALERT_DISTANCE_M
    near = config.CRITICAL_DISTANCE_M
    if far <= near:
        return 1.0
    return max(0.0, min(1.0, (far - distance) / (far - near)))


def map_obstacles(obstacles, t, goal_bearing=None, motor_angles=None):
    """Ana esleme fonksiyonu.

    obstacles    : [(kerteriz, mesafe), ...] govde cercevesinde
    t            : simdiki zaman (saniye) - nabiz zarflari icin
    goal_bearing : hedefin kerterizi (radyan) ya da None
    motor_angles : motor acilari; None ise config'ten alinir

    Doner: MotorCommand listesi
    """
    if motor_angles is None:
        motor_angles = config.MOTOR_ANGLES_RAD
    n = len(motor_angles)

    cmds = [MotorCommand(i) for i in range(n)]

    # --- 1) Acil durum her seyi ezer -------------------------------------
    critical = [o for o in obstacles if o[1] <= config.CRITICAL_DISTANCE_M]
    if critical:
        env = patterns.emergency_envelope(t)
        for c in cmds:
            c.intensity = env
            c.pattern = patterns.EMERGENCY
        return _apply_floor(cmds)

    # --- 2) Engeller ------------------------------------------------------
    any_obstacle = False
    for bearing, dist in obstacles:
        if dist > config.ALERT_DISTANCE_M:
            continue
        any_obstacle = True
        prox = proximity_of(dist)
        env = patterns.obstacle_envelope(t, prox)
        for i, w in enumerate(_spread_weights(bearing, motor_angles)):
            if w <= 0.0:
                continue
            value = env * w
            # Birden fazla engel ayni motora dusuyorsa en siddetlisi kazanir
            if value > cmds[i].intensity:
                cmds[i].intensity = value
                cmds[i].pattern = patterns.OBSTACLE

    # --- 3) Yonlendirme ---------------------------------------------------
    # Engelin oldugu motora yonlendirme koymayiz: "oraya git" ile
    # "orada engel var" ayni motorda cakisirsa kullanici kafasi karisir.
    if goal_bearing is not None:
        env = patterns.guidance_envelope(t)
        for i, w in enumerate(_spread_weights(goal_bearing, motor_angles)):
            if w <= 0.0:
                continue
            if cmds[i].pattern == patterns.OBSTACLE:
                continue
            value = env * w
            if value > cmds[i].intensity:
                cmds[i].intensity = value
                cmds[i].pattern = patterns.GUIDANCE

    # --- 4) Soyleyecek bir sey yoksa "yasiyorum" nabzi --------------------
    if not any_obstacle and goal_bearing is None:
        hb = patterns.heartbeat_envelope(t)
        if hb > 0.0:
            for c in cmds:
                c.intensity = hb
                c.pattern = patterns.HEARTBEAT

    return _apply_floor(cmds)


def _apply_floor(cmds):
    """Hissedilmeyecek kadar zayif komutlari sifirla.

    Gercek motorlar da bu esigin altinda donmez; simulasyonun
    donanimla ayni davranmasi icin burada da uyguluyoruz.
    """
    for c in cmds:
        if c.intensity < config.MOTOR_MIN_INTENSITY:
            c.intensity = 0.0
            if c.pattern != patterns.IDLE:
                c.pattern = patterns.IDLE
        elif c.intensity > 1.0:
            c.intensity = 1.0
    return cmds
