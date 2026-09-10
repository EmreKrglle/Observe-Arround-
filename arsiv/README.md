# Arşiv — SEE (Smart Environment Explorer)

Projenin ilk sürümü: kamera tabanlı nesne algılama, insan pozu tahmini ve
stereo derinlik haritası. Aktif geliştirme `faz0/` altında devam ediyor;
bu kod referans olarak saklanıyor ve çalışır durumda.

## Özellikler
- **Nesne Algılama & Yönlendirme**: YOLOv8 Segmentation & Tracking
- **Pose Estimation**: YOLOv8 Pose modeli ile insan pozu çıkarımı
- **Derinlik Haritalama**: Stereo kamera ile mesafe ölçümü
- **GUI**: Tkinter kontrol paneli ve yönlendirme gösterimi

## Proje Yapısı
```
arsiv/
├── requirements.txt
└── src/
    ├── main.py            # Giriş noktası
    ├── gui/
    │   └── app.py         # Tkinter arayüzü
    └── vision/
        ├── detector.py    # Nesne algılama ve takip (YOLOv8-seg)
        ├── pose.py        # İnsan pozu tahmini (YOLOv8-pose)
        ├── depth.py       # Stereo derinlik haritası
        └── utils.py       # Kamera yardımcıları ve kamera index ayarları
```

## Kurulum ve Çalıştırma
```bash
pip install -r arsiv/requirements.txt
python arsiv/src/main.py
```

YOLO modelleri (`yolov8n-seg.pt`, `yolov8n-pose.pt`) ilk kullanımda otomatik indirilir.

## Kamera Ayarları
Kamera index'leri `src/vision/utils.py` içinde tanımlıdır:
- `CAMERA_INDEX`: Nesne algılama ve pose için kamera
- `STEREO_LEFT_INDEX` / `STEREO_RIGHT_INDEX`: Derinlik haritası için stereo kamera çifti

Pencerelerden çıkmak için: algılama ve pose'da `q`, derinlik haritasında `ESC`.
