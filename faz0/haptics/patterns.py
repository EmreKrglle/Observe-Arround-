"""
Titresim kaliplari.

Uc kalip var ve hepsinin amaci birbirinden AYIRT EDILEBILIR olmak:

  yonlendirme : yumusak, yavas, kisa dokunuslar   -> "bu tarafa git"
  engel       : sert, yaklastikca hizlanan nabiz  -> "burada bir sey var"
  acil        : tum bant birden, en yuksek siddet -> "hemen dur"

Ayni motor konumu iki zit anlam tasiyabilir ("sola git" ve "solda engel").
Bu yuzden YON motorun yeriyle, ANLAM ritimle kodlanir.

Buradaki sayilar bir baslangic tahminidir, dogrulanmis DEGILDIR.
Gercek gonullulerle test edip config.py'den degistireceksiniz.
"""

import math

import config


def _pulse(t, hz, duty):
    """Kare dalga nabiz zarfi.

    t   : saniye
    hz  : saniyedeki nabiz sayisi
    duty: nabzin acik kaldigi oran (0..1)

    Doner: 0.0 (kapali) veya 1.0 (acik)
    """
    if hz <= 0.0:
        return 1.0
    phase = (t * hz) % 1.0
    return 1.0 if phase < duty else 0.0


def guidance_envelope(t):
    """Yonlendirme kalibi: yumusak ve yavas."""
    p = config.PATTERN_GUIDANCE
    return p["amplitude"] * _pulse(t, p["pulse_hz"], p["duty"])


def obstacle_envelope(t, proximity):
    """Engel kalibi.

    proximity: 0.0 (uyari mesafesinin sinirinda) .. 1.0 (burnumuzun dibinde)

    Yaklastikca hem hizlanir hem siddetlenir - park sensoru mantigi.
    Bu, kullanicinin mesafeyi ogrenmesini saglar.
    """
    p = config.PATTERN_OBSTACLE
    proximity = max(0.0, min(1.0, proximity))
    hz = p["pulse_hz_far"] + (p["pulse_hz_near"] - p["pulse_hz_far"]) * proximity
    amp = p["amplitude_far"] + (p["amplitude_near"] - p["amplitude_far"]) * proximity
    return amp * _pulse(t, hz, p["duty"])


def emergency_envelope(t):
    """Acil kalibi: en yuksek siddet, tum bant."""
    p = config.PATTERN_EMERGENCY
    return p["amplitude"] * _pulse(t, p["pulse_hz"], p["duty"])


def heartbeat_envelope(t):
    """'Yasiyorum' nabzi.

    Sistem calisiyor ama soyleyecek bir sey yok. Cok hafif, seyrek.

    Neden gerekli: sessiz bir bant ile OLU bir bant, kullanici icin
    ayirt edilemez. Kullanici sessizligi "onum temiz" diye okur.
    Bu nabiz olmadan, pili biten cihaz kullaniciyi yanlis guvene sokar.
    """
    period = config.HEARTBEAT_PERIOD_S
    if period <= 0.0:
        return 0.0
    phase = t % period
    if phase < config.HEARTBEAT_DURATION_S:
        # Yumusak giris-cikis (ani baslamasin, dikkat dagitmasin)
        k = phase / config.HEARTBEAT_DURATION_S
        return config.HEARTBEAT_AMPLITUDE * math.sin(math.pi * k)
    return 0.0


# Kalip isimleri - ekranda ve kayitta kullanilir
GUIDANCE = "yonlendirme"
OBSTACLE = "engel"
EMERGENCY = "acil"
HEARTBEAT = "kalp-atisi"
IDLE = "bos"
