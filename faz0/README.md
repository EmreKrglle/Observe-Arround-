# Faz 0 — Titreşimli yön bandı simülatörü

Görme engelli kullanıcılar için gövdeye takılan titreşimli yönlendirme
bandının 2B simülatörü. Donanım olmadan çalışır.

**Bağımlılık yok.** Sadece Python 3.8+ ve standart kütüphane (tkinter).

```bash
python3 run_sim.py                      # koridor senaryosu
python3 run_sim.py --scenario dar_kapi
python3 run_sim.py --headless 30        # ekransız, puan basar
python3 -m unittest discover tests      # 54 test
```

Kontroller: yön tuşları yürü/dön · `Q`/`E` yana adım · `[`/`]` güneş ışığı ·
`G` hedef rehberi · `R` baştan · `boşluk` duraklat · `ESC` çık

---

## Bu simülatör ne işe yarıyor

Amacı geometriyi ve karar mantığını donanım almadan doğrulamak:

- Engel 45 derecede ise doğru motor mu titriyor?
- Sol/sağ işaretleri karışmış mı?
- Üç titreşim kalıbı doğru tetikleniyor mu?
- Kaç sensör gerekiyor?

## Bu simülatör ne işe YARAMIYOR

Bu, en önemli bölüm. Sanal sensör modeli **gerçeğinden iyimserdir** ve
şu problemleri hiç taşımaz:

| Gerçek dünya problemi | Simülatörde var mı? |
|---|---|
| Islak zemin / su birikintisi ayna gibi yansıtır | ❌ yok |
| Siyah mont menzili ~4'e böler | ❌ yok |
| Cam kapı neredeyse görünmez | ❌ yok |
| Lense konan damla "2 cm'de nesne" der | ❌ yok |
| Sensör kiri, buğulanma, parmak izi | ❌ yok |
| **Kullanıcı titreşimi doğru anlıyor mu?** | ❌ **yok** |

> **Simülasyonda %100 başarı, sahada çalışacağı anlamına gelmez.**
> Simülatör geometriyi doğrular, sensör fiziğini ve insan algısını değil.

`config.py` içindeki gürültü ve menzil değerlerini **ağırlaştırıp** sistemin
ne kadar dayanıklı olduğunu görün. İyimser ayarlarla test etmek kendinizi
kandırmaktır.

---

## Simülatörün bulduğu tasarım hatası

Daha ilk çalıştırmada, planlanan **6 sensörlü** tasarımın ciddi bir kör
noktası olduğu ortaya çıktı.

VL53L1X'in görüş açısı 27°. 6 sensör 60° aralıklarla dizilirse:

| Sensör | Aralık | Kapsama | Kör boşluk |
|---|---|---|---|
| **6** | 60° | **%45** | **33°** |
| 8 | 45° | %60 | 18° |
| 12 | 30° | %90 | 3° |
| 16 | 22.5° | %100 | 0° |

En kötüsü: **tam yanlar (±90°) kör boşluğa denk geliyor** — ve koridor
duvarları tam olarak orada.

Simülasyonda ölçülen sonuç (koridor senaryosu, 25 sn):

| Sensör | Kapsama | Çarpışma | Kaçırma |
|---|---|---|---|
| 6 | %45 | 1 | %47.9 |
| 8 | %60 | 0 | %31.1 |
| 12 | %90 | 0 | %20.9 |
| 16 | %100 | 0 | %8.9 |

**Sonuç:** 6 sensör yetmiyor. Seçenekler:

1. Sensör sayısını artır (maliyet: her biri ~$6)
2. Sensörleri öne yoğunlaştır — arka taraf daha az kritik, yürüdüğünüz
   yön daha önemli
3. Daha geniş görüş açılı sensör kullan (VL53L5CX 45°, ama ~3 kat pahalı)

**Öneri: 8 sensörü öne yoğunlaştırın** (ön 180°'ye 6 tane, arkaya 2 tane).
Bu, 16 sensörün maliyeti olmadan kritik bölgede tam kapsama verir.

> Bu bulgu, simülatörün maliyetini ilk gün çıkardı: donanım alınsaydı
> aynı ders ~$40 ve iki hafta ederdi.

Bir diğer bulgu: **açık alanda yalancı alarm yok.** Yürüyüş salınımını
15°'ye kadar abartsak bile boş odada bant sessiz kalıyor.

---

## Dosya düzeni

```
faz0/
├── config.py            TÜM ayarlar burada. Başka yerde sabit sayı yok.
├── run_sim.py           ana program (grafik + ekransız mod)
├── sim/
│   ├── geometry.py      saf matematik — ışın atma, açı dönüşümleri
│   ├── world.py         duvarlar, direkler, hareketli engeller
│   ├── sensors.py       sanal ToF (idealize — sınırları dosyada yazılı)
│   ├── walker.py        yürüyen kullanıcı + yürüyüş salınımı
│   ├── scoring.py       çarpışma / yalancı alarm / kaçırma sayacı
│   └── render.py        tkinter çizim
├── haptics/             ⭐ simülatörden BAĞIMSIZ — donanımda aynen çalışır
│   ├── patterns.py      üç titreşim kalıbı
│   ├── mapper.py        engel listesi → motor komutları
│   └── backends.py      sim / konsol / gerçek donanım (PCA9685)
├── scenarios/           JSON senaryolar
└── tests/               54 birim testi
```

`haptics/` klasörünün simülatörden bağımsız olması bilinçli:
**simülasyonda doğruladığınız mantık, banda taktığınızda değişmez.**

---

## Donanım-döngüde test (projenin en değerli kullanımı)

Sanal dünya + **gerçek** motorlar + **gerçek** insan:

```bash
python3 run_sim.py --backend hardware --scenario koridor
```

Gözü bağlı bir gönüllü bandı takar, siz klavyeden yürütürsünüz.

Neden değerli:
- Simüle edilemeyen tek şey (insan algısı) gerçek kalır
- Senaryolar tekrarlanabilir — 20 gönüllüye aynı koridoru yaşatabilirsiniz
- Merdiven/trafik riski yok, kişi odada duruyor
- Sensör almanıza gerek yok, sadece ~$25'lık motor bandı

Kablolamayı doğrulamak için önce: `python3 run_sim.py --test-motors`

### Seviye 0 devresi

```
Raspberry Pi / ESP32 ──I2C──> PCA9685 ──> ULN2803A ──> 6× LRA motor
```

`pip install smbus2` gerekir (sadece `--backend hardware` için).

---

## Titreşim kalıpları

Aynı motor iki zıt anlam taşıyabilir ("sola git" / "solda engel").
Bu yüzden **yön motorun yeriyle, anlam ritimle** kodlanır.

| Kalıp | Ritim | Şiddet | Anlamı |
|---|---|---|---|
| yönlendirme | 2 Hz, kısa dokunuş | 0.55 | "bu tarafa git" |
| engel | 2→9 Hz (yaklaştıkça hızlanır) | 0.45→1.0 | "burada bir şey var" |
| acil | 6 Hz, **tüm bant** | 1.0 | "hemen dur" |
| kalp atışı | 30 sn'de bir, çok hafif | 0.18 | "sistem çalışıyor" |

**Kalp atışı neden var:** sessiz bir bant ile *ölü* bir bant, kullanıcı için
ayırt edilemez. Kullanıcı sessizliği "önüm temiz" diye okur. Bu nabız
olmadan pili biten cihaz kullanıcıyı yanlış güvene sokar.

⚠️ Bu değerler **doğrulanmış değildir** — bir başlangıç tahminidir.
Gerçek gönüllülerle test edip `config.py`'den değiştireceksiniz.

---

## Başarı kriterleri

| Ölçüt | Hedef |
|---|---|
| Çarpışma | 0 |
| Yalancı alarm | < 1/dakika |
| Kaçırılan engel | < %5 |

Yalancı alarm en önemlisidir: sürekli boşuna titreyen bir bant, hiç
titremeyen bir banttan daha çabuk çöpe atılır.

---

## Sonraki adımlar

1. Sensör yerleşimini yeniden tasarla (yukarıdaki bulgu)
2. Titreşim kalıplarını gözü bağlı gönüllülerle test et
3. `--backend hardware` yolunu gerçek donanımda doğrula *(bu kod
   donanım olmadan yazıldı, test edilmedi)*
4. Görme engelli gönüllülerle test — **etik kurul onayı gerekir**

### Test güvenlik kuralları

- İç mekân, düz zemin, merdiven ve trafik yok
- Test eden kişinin **bastonu elinde**
- Bir gözlemci daima dokunma mesafesinde
- Faz 0 cihazına asla tek başına yürüme izni verilmez

**Bu sistem beyaz bastonun yerine geçmez, üstüne biner.**
