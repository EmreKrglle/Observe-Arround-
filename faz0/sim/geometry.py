"""
Saf geometri yardimcilari.

Bu dosyada ekran, sensor ya da motor yoktur - sadece matematik.
Bu sayede birim testleriyle dogrulanabilir. Simulatorun geri kalani
yanlis olsa bile buradaki fonksiyonlar dogruysa acilar dogrudur.

Koordinat sistemi
-----------------
Dunya:  x sag, y yukari (matematik standardi)
Aci:    radyan, 0 = +x yonu, pozitif = saat yonunun TERSI
Govde:  0 = tam on, pozitif aci = SOL taraf
"""

import math

TWO_PI = 2.0 * math.pi


def normalize_angle(a: float) -> float:
    """Aciyi (-pi, pi] araligina indirger.

    Iki aciyi karsilastirmadan once daima bunu kullanin; yoksa
    350 derece ile 10 derece arasindaki fark 340 derece cikar.
    """
    a = math.fmod(a + math.pi, TWO_PI)
    if a <= 0.0:
        a += TWO_PI
    return a - math.pi


def angle_diff(a: float, b: float) -> float:
    """a ile b arasindaki en kisa acisal fark, (-pi, pi]."""
    return normalize_angle(a - b)


def ray_circle(ox, oy, dx, dy, cx, cy, r):
    """Isin-daire kesisimi.

    Isin (ox, oy) noktasindan (dx, dy) BIRIM yonunde gider.
    Daire merkezi (cx, cy), yaricap r.

    En yakin pozitif t degerini dondurur (metre cinsinden mesafe),
    kesisim yoksa None.
    """
    lx = cx - ox
    ly = cy - oy
    # Isin uzerinde daire merkezine en yakin noktanin parametresi
    tca = lx * dx + ly * dy
    d2 = (lx * lx + ly * ly) - tca * tca
    r2 = r * r
    if d2 > r2:
        return None
    thc = math.sqrt(r2 - d2)
    t0 = tca - thc
    t1 = tca + thc
    # Isinin onunde kalan en yakin kesisim
    if t0 >= 0.0:
        return t0
    if t1 >= 0.0:
        return t1          # isin dairenin icinden basliyor
    return None


def ray_segment(ox, oy, dx, dy, ax, ay, bx, by):
    """Isin-dogru parcasi kesisimi.

    Isin (ox, oy) -> (dx, dy) birim yon.
    Parca A(ax, ay) -> B(bx, by).

    Kesisim mesafesini dondurur, yoksa None.
    """
    # v1 = O - A, v2 = B - A, v3 = dik(D)
    v1x = ox - ax
    v1y = oy - ay
    v2x = bx - ax
    v2y = by - ay
    v3x = -dy
    v3y = dx

    denom = v2x * v3x + v2y * v3y
    if abs(denom) < 1e-12:
        return None                      # isin parcaya paralel

    t = (v2x * v1y - v2y * v1x) / denom   # isin uzerindeki mesafe
    s = (v1x * v3x + v1y * v3y) / denom   # parca uzerindeki oran 0..1

    if t >= 0.0 and 0.0 <= s <= 1.0:
        return t
    return None


def point_segment_distance(px, py, ax, ay, bx, by):
    """Bir noktanin dogru parcasina en kisa uzakligi.

    Carpisma kontrolu icin kullanilir.
    """
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    seg_len2 = vx * vx + vy * vy
    if seg_len2 < 1e-12:
        return math.hypot(wx, wy)         # parca aslinda bir nokta
    # Noktanin parca uzerindeki izdusumu, 0..1 arasina kirpilir
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg_len2))
    cx = ax + t * vx
    cy = ay + t * vy
    return math.hypot(px - cx, py - cy)


def to_body_frame(px, py, ox, oy, heading):
    """Dunya koordinatindaki bir noktayi govde cercevesine cevirir.

    Donen deger: (mesafe, kerteriz)
      mesafe  - metre
      kerteriz - radyan, 0 = tam on, pozitif = sol

    TF2'nin gercek sistemde yapacagi isin elle yazilmis hali.
    """
    dx = px - ox
    dy = py - oy
    dist = math.hypot(dx, dy)
    bearing = normalize_angle(math.atan2(dy, dx) - heading)
    return dist, bearing
