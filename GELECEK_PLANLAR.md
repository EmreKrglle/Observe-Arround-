# Gelecek Planları

Henüz başlanmamış, ileriki fazlar için değerlendirilmiş fikirler.
Güncel açık sorunlar için bkz. [faz0/README.md](faz0/README.md#açık-kalan-sorunlar).

---

## ROS 2 entegrasyonu

**Durum:** Değerlendirildi, ertelendi. **Ne zaman:** Faz 1, gerçek donanım
geldiğinde ve birden fazla sensör türü (ToF + kamera + radar) birleştirilmeye
başlandığında.

### Neden uygun

Kodun ana sınırı zaten ROS 2'nin topic mantığına uyuyor.
`haptics/mapper.py` sensörleri bilmez; sadece `[(kerteriz, mesafe), ...]`
listesi alıp motor komutu üretir. Radar ve kamera da aynı listeye katkı
yapabilir.

### Önerilen node yapısı

```
tof_node          ──/obstacles──>  haptics_node  ──/motor_cmd──>  motor_driver_node
(VL53L1X × 10)                     (mapper.py)                    (PCA9685)
camera_node   ──┘                       ↑
radar_node    ──┘               /goal_bearing (navigasyon)
```

- Sensörler: `sensor_msgs/Range` (sensör başına) ya da `sensor_msgs/LaserScan` (tüm dizi)
- `haptics/` kodu değişmeden bir `rclpy` node'una sarılır
- Motor sürücü node'u mevcut `hardware` backend'inin işini yapar
- Eski YOLO kodu (`arsiv/`) bir `camera_node` olarak geri dönebilir

### Kazançlar

- **rosbag:** gerçek sensör verisini kaydedip masada tekrar oynatmak (gönüllü testleri için)
- **RViz:** sensör ve motor durumunu canlı izlemek
- **Sensör birleştirme:** yeni sensör türü = yeni node
- **Gazebo:** 3B simülasyon; merdiven, masa altı gibi 2B'de modellenemeyen engeller

### Maliyetler ve riskler

- **Windows:** ROS 2 desteği zayıf; WSL2 ya da Docker ile Ubuntu gerekir
- **Donanım:** Raspberry Pi'de rahat çalışır (Ubuntu 24.04 + ROS 2 Jazzy).
  ESP32'de tam ROS 2 çalışmaz; **micro-ROS** (C/C++ firmware + Pi'de agent) gerekir
- **Bağımlılık:** Faz 0 bilinçli olarak bağımlılıksız; ROS 2 kurulumu saatler alır
- **Şu anki sorunları çözmez:** yalancı alarm ve güneşte menzil sorunları karar
  mantığında ve sensör seçiminde

### Uygulama ilkesi

Çekirdek (`sim/`, `haptics/`) saf Python kalır. ROS 2, yanına eklenen ayrı
ve ince bir `ros2/` paketi olur; sadece mevcut kodu node'lara sarar.
Simülatör ROS 2 olmadan çalışmaya devam eder.

### İlk adım (hazır olunduğunda)

Simülatörü değiştirmeden `/obstacles` ve `/motor_cmd` topic'lerini yayınlayan
tek bir köprü node'u.

### Önce netleşmesi gerekenler

- Kontrolcü seçimi: Raspberry Pi mi, ESP32 mi? (ESP32 normal Python da
  çalıştıramaz; sadece MicroPython)
- `haptics/backends.py` içindeki donanım kodunun node'a nasıl sarılacağı
  (henüz incelenmedi)
