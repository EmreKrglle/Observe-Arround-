"""
Faz 0 simulatoru - tum ayarlanabilir parametreler.

Bu dosyadaki sayilari degistirerek sistemin davranisini deneyebilirsiniz.
Kodun baska hicbir yerinde sabit sayi yoktur; hepsi buradan gelir.
"""

import math

# ---------------------------------------------------------------------------
# TITRESIM MOTORLARI
# ---------------------------------------------------------------------------

# Motorlarin govde uzerindeki acilari (derece).
# 0 = tam on, pozitif yon = SOL (saat yonunun tersi), 180 = tam arka.
# 6 motor, 60 derece araliklarla.
MOTOR_ANGLES_DEG = [0, 60, 120, 180, 240, 300]

MOTOR_NAMES = ["on", "on-sol", "arka-sol", "arka", "arka-sag", "on-sag"]

# Bir motorun hissedilebilir en dusuk siddeti (0-1).
# Bunun altindaki komutlar sifira yuvarlanir - gercek motorlar da zaten donmez.
MOTOR_MIN_INTENSITY = 0.12

# Iki motor arasina dusen bir yon icin siddeti ikisine paylastir.
# Insan bunu "iki motorun arasinda tek bir nokta" olarak algilar
# (hayalet dokunus / funneling illusion). Kapatirsaniz en yakin
# tek motor calisir ve yon cozunurlugu 60 dereceye duser.
USE_PHANTOM_INTERPOLATION = True


# ---------------------------------------------------------------------------
# SANAL ToF SENSORLERI
# ---------------------------------------------------------------------------

# Sensor acilari (derece). Varsayilan olarak motorlarla ayni yerlerde.
# Gercek bir tasarimda one daha fazla sensor koymak mantikli olur -
# bu listeyi degistirerek deneyebilirsiniz.
SENSOR_ANGLES_DEG = [0, 60, 120, 180, 240, 300]

# Her sensorun gorus konisi (derece). VL53L1X ~27 derece.
SENSOR_FOV_DEG = 27.0

# Koniyi kac isinla ornekleyelim (tek sayi olmali, ortada bir isin olsun).
SENSOR_RAYS = 7

# Ideal kosullarda azami menzil (metre). VL53L1X karanlikta ~4 m.
SENSOR_MAX_RANGE_M = 4.0

# Asgari menzil - bunun altini sensor guvenilir okumaz.
SENSOR_MIN_RANGE_M = 0.05

# Olcum gurultusu: mesafeyle orantili standart sapma (oran).
# 0.02 = 2 metrede ~4 cm gurultu.
SENSOR_NOISE_RATIO = 0.02

# Her okumada "hic cevap gelmeme" olasiligi (0-1).
SENSOR_DROPOUT_PROB = 0.01


# ---------------------------------------------------------------------------
# ORTAM KOSULLARI  ("gunes kaydiragi")
# ---------------------------------------------------------------------------
# 0.0 = ic mekan / karanlik  ->  menzil tam
# 1.0 = direkt ogle gunesi   ->  menzil AMBIENT_WORST_RATIO'ya duser
#
# Simulasyon sirasinda [ ve ] tuslariyla degistirilebilir.
# ---------------------------------------------------------------------------

AMBIENT_LIGHT = 0.0
AMBIENT_WORST_RATIO = 0.30   # gunesde menzil %30'e iner


# ---------------------------------------------------------------------------
# KULLANICI (yuruyen kisi)
# ---------------------------------------------------------------------------

WALK_SPEED_MPS = 1.2          # normal yuruyus hizi
TURN_SPEED_DPS = 90.0         # yerinde donus hizi (derece/saniye)
BODY_RADIUS_M = 0.25          # carpisma yaricapi

# Yuruyus salinimi: govde her adimda saga sola hafifce doner.
# Gercek hayatta bu, sensorlerin nereye baktigini surekli degistirir ve
# yalanci alarmlarin baslica sebebidir. 0 yaparsaniz sorun kaybolur -
# ama gercek dunyada kaybolmaz.
GAIT_SWAY_DEG = 3.0
GAIT_FREQ_HZ = 2.0


# ---------------------------------------------------------------------------
# UYARI ESIKLERI
# ---------------------------------------------------------------------------

ALERT_DISTANCE_M = 2.0        # bu mesafeden itibaren engel bildirilir
CRITICAL_DISTANCE_M = 0.6     # bu mesafede acil durum kalibi devreye girer

# Bir engel kac kare ust uste gorulmezse unutulsun.
# Tek karelik gurultuyu susturur.
OBSTACLE_MEMORY_FRAMES = 3


# ---------------------------------------------------------------------------
# TITRESIM KALIPLARI
# ---------------------------------------------------------------------------
# Uc kalibin da amaci birbirinden AYIRT EDILEBILIR olmak.
# Bu degerleri gercek gonullulerle test edip degistireceksiniz -
# buradaki sayilar bir baslangic tahminidir, dogrulanmis degildir.
# ---------------------------------------------------------------------------

PATTERN_GUIDANCE = {
    "name": "yonlendirme",
    "pulse_hz": 2.0,      # saniyede 2 nabiz
    "duty": 0.25,         # nabzin acik kaldigi oran (kisa dokunus)
    "amplitude": 0.55,    # yumusak
}

PATTERN_OBSTACLE = {
    "name": "engel",
    "pulse_hz_far": 2.0,   # uzaktayken yavas
    "pulse_hz_near": 9.0,  # yaklastikca hizlanir (park sensoru mantigi)
    "duty": 0.45,
    "amplitude_far": 0.45,
    "amplitude_near": 1.0,
}

PATTERN_EMERGENCY = {
    "name": "acil",
    "pulse_hz": 6.0,
    "duty": 0.6,
    "amplitude": 1.0,
    "all_motors": True,    # tum bant birden titrer
}

# Sistem calisiyor ama soyleyecek bir sey yok: her N saniyede bir
# cok hafif "yasiyorum" nabzi. Sessiz bant ile olu bandi ayirmak icin.
HEARTBEAT_PERIOD_S = 30.0
HEARTBEAT_AMPLITUDE = 0.18
HEARTBEAT_DURATION_S = 0.12


# ---------------------------------------------------------------------------
# SIMULASYON / EKRAN
# ---------------------------------------------------------------------------

FPS = 30
PIXELS_PER_METER = 55
WINDOW_W = 1100
WINDOW_H = 760

# Renkler (koyu tema)
COL_BG = "#12141a"
COL_GRID = "#1e222c"
COL_WALL = "#4a5163"
COL_PILLAR = "#5b6478"
COL_MOVER = "#8b5a3c"
COL_GOAL = "#2e7d5b"
COL_BODY = "#e8eaf0"
COL_RAY = "#2a3550"
COL_HIT = "#c9683f"
COL_TEXT = "#c8cdd8"
COL_DIM = "#6b7280"
COL_WARN = "#d4a04a"
COL_CRIT = "#cf5b52"


def sensor_max_range(ambient_light: float) -> float:
    """Ortam isigina gore gercekci azami menzil.

    ambient_light 0..1 arasi. Dogrusal bir yaklasim - gercek sensorlerde
    dusus daha karmasik ama simulasyon icin yeterli.
    """
    ratio = 1.0 - (1.0 - AMBIENT_WORST_RATIO) * max(0.0, min(1.0, ambient_light))
    return SENSOR_MAX_RANGE_M * ratio


MOTOR_ANGLES_RAD = [math.radians(a) for a in MOTOR_ANGLES_DEG]
SENSOR_ANGLES_RAD = [math.radians(a) for a in SENSOR_ANGLES_DEG]
