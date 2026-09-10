"""
Motor arka uclari.

Ayni MotorCommand listesi hem ekrana hem gercek motorlara gidebilir.
Ust katman (mapper) hangisini kullandigini bilmez - bu sayede
simulasyonda dogruladiginiz mantik banda takildiginda degismez.

  SimBackend       : sadece hafizada tutar, ekran cizer
  ConsoleBackend   : terminale yazar (hata ayiklama)
  PCA9685Backend   : GERCEK donanim - Seviye 0 bandi
"""

import config


class Backend:
    """Ortak arayuz."""

    def send(self, cmds):
        raise NotImplementedError

    def stop(self):
        """Tum motorlari sustur. Cikista MUTLAKA cagrilmali."""
        pass

    def close(self):
        self.stop()


class SimBackend(Backend):
    """Simulasyon: komutlari sakla, cizim bunlari okusun."""

    def __init__(self):
        self.last = []

    def send(self, cmds):
        self.last = cmds

    def stop(self):
        self.last = []


class ConsoleBackend(Backend):
    """Terminale cubuk grafik basar. Ekransiz hata ayiklama icin."""

    def __init__(self, every=10):
        self.every = every
        self._n = 0

    def send(self, cmds):
        self._n += 1
        if self._n % self.every:
            return
        parts = []
        for c in cmds:
            bars = int(c.intensity * 8)
            parts.append(f"{config.MOTOR_NAMES[c.index]:>9}|{'#' * bars:<8}|")
        print(" ".join(parts))

    def stop(self):
        print("[motorlar durduruldu]")


class PCA9685Backend(Backend):
    """GERCEK DONANIM - PCA9685 PWM surucusu uzerinden LRA motorlar.

    Devre (Seviye 0, ~$25):

        Raspberry Pi / ESP32
             |  I2C (SDA, SCL)
        PCA9685  (16 kanal PWM, adres 0x40)
             |  kanal 0..5
        ULN2803A (darlington dizi - motorlari surecek akimi saglar)
             |
        6 adet LRA motor

    LRA motorlar rezonans frekanslarinda (tipik ~175 Hz) surulur.
    PWM tasiyici frekansini buna ayarliyor, siddeti doluluk orani
    (duty cycle) ile veriyoruz.

    NOT: Bu yontem DRV2605L surucu cipinin yaptigi otomatik rezonans
    takibi ve aktif frenlemeyi yapmaz. Titresim biraz daha yumusak
    olur. Seviye 0 icin kabul edilebilir; kaliplarin ayirt edilebilirligi
    yetersiz cikarsa ilk supheleneceginiz yer burasi olsun.

    !!! BU KOD DONANIM OLMADAN TEST EDILEMEDI !!!
    Ilk calistirdiginizda motorlari tek tek deneyin (--test-motors).
    """

    MODE1 = 0x00
    PRESCALE = 0xFE
    LED0_ON_L = 0x06

    def __init__(self, address=0x40, bus_number=1, lra_hz=175.0):
        try:
            from smbus2 import SMBus
        except ImportError as exc:
            raise RuntimeError(
                "Gercek donanim icin 'smbus2' gerekli:  pip install smbus2\n"
                "Donaniminiz yoksa --hardware bayragini kullanmayin."
            ) from exc

        self.address = address
        self.bus = SMBus(bus_number)
        self._set_pwm_freq(lra_hz)
        self.stop()

    def _write8(self, reg, value):
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def _set_pwm_freq(self, hz):
        # PCA9685 dahili osilatoru 25 MHz, 12 bit cozunurluk
        prescale = int(round(25_000_000.0 / (4096.0 * hz)) - 1)
        prescale = max(3, min(255, prescale))
        old_mode = self.bus.read_byte_data(self.address, self.MODE1)
        self._write8(self.MODE1, (old_mode & 0x7F) | 0x10)   # uyku moduna al
        self._write8(self.PRESCALE, prescale)
        self._write8(self.MODE1, old_mode)                    # uyandir
        # Osilatorun oturmasi icin kisa bekleme
        import time
        time.sleep(0.005)
        self._write8(self.MODE1, old_mode | 0xA0)             # restart + autoinc

    def _set_channel(self, channel, duty):
        """duty 0.0 .. 1.0"""
        off = int(max(0.0, min(1.0, duty)) * 4095)
        base = self.LED0_ON_L + 4 * channel
        self.bus.write_byte_data(self.address, base + 0, 0)
        self.bus.write_byte_data(self.address, base + 1, 0)
        self.bus.write_byte_data(self.address, base + 2, off & 0xFF)
        self.bus.write_byte_data(self.address, base + 3, off >> 8)

    def send(self, cmds):
        for c in cmds:
            self._set_channel(c.index, c.intensity)

    def stop(self):
        for ch in range(len(config.MOTOR_ANGLES_DEG)):
            try:
                self._set_channel(ch, 0.0)
            except OSError:
                pass

    def close(self):
        self.stop()
        try:
            self.bus.close()
        except Exception:
            pass


def make_backend(kind):
    """Isimden arka uc uretir."""
    if kind == "sim":
        return SimBackend()
    if kind == "console":
        return ConsoleBackend()
    if kind == "hardware":
        return PCA9685Backend()
    raise ValueError(f"bilinmeyen arka uc: {kind}")
