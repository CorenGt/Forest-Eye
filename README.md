# 🌲 ForestEye - Orman Segmentasyon Projesi

![Predict]('ss_1.png')

## 📖 Proje Açıklaması

ForestEye, satellit görüntülerinden orman alanlarını otomatik olarak tespit eden ve segmentasyon yapan yapay zeka tabanlı bir uygulamadır. Proje, U-Net derin öğrenme mimarisi kullanarak yüksek doğrulukla orman ve çorak arazi alanlarını ayırt edebilmektedir.

## 🎯 Proje Amacı

- Satellit görüntülerinden orman alanlarını otomatik tespit
- Orman ve çorak toprak alanlarının hassas segmentasyonu
- Çevre koruma ve orman yönetimi için karar destek sistemi
- Ormansızlaşma takibi ve analizi

## 🚀 Özellikler

- **Gerçek Zamanlı Segmentasyon**: Yüklenen satellit görüntülerini anında analiz eder
- **Görsel Sonuçlar**: Orijinal görüntü ile segmentasyon sonucunu karşılaştırmalı gösterir
- **Güven Oranı**: Modelin tahmini için güven skorunu görselleştirir
- **Kullanıcı Dostu Arayüz**: CustomTkinter ile modern desktop arayüzü
- **Yüksek Doğruluk**: U-Net + ResNet34 mimarisi ile optimize edilmiş performans

## 🛠️ Kullanılan Teknolojiler

### Derin Öğrenme & AI
- **TensorFlow/Keras**: Ana derin öğrenme framework'ü
- **U-Net**: Segmentasyon için özel tasarlanmış mimari
- **ResNet34**: Encoder omurgası olarak kullanılan pre-trained model
- **Segmentation Models**: Segmentasyon modelleri kütüphanesi

### Görüntü İşleme
- **OpenCV**: Görüntü ön işleme ve manipülasyon
- **PIL (Pillow)**: Görüntü yükleme ve dönüştürme
- **NumPy**: Numerik hesaplamalar

### Veri Görselleştirme
- **Matplotlib**: Grafik ve görselleştirme
- **CustomTkinter**: Modern desktop arayüzü

### Veri Yönetimi
- **Pandas**: Veri analizi ve manipülasyon
- **Kaggle API**: Veri seti indirme

## 📊 Model Detayları

- **Mimari**: U-Net with ResNet34 encoder
- **Giriş Boyutu**: 256x256x3 (RGB görüntü)
- **Çıkış**: 256x256x1 (Binary segmentasyon maskesi)
- **Optimizasyon**: Adam optimizer (lr=1e-4)
- **Loss Function**: BCE + Jaccard Loss
- **Metrikler**: IoU Score, F1-Score, Binary Accuracy

![Result]('ss_2.png')

## 📁 Dosya Yapısı

```
ForestEye/
├── app.py                    # Ana desktop uygulaması
├── foresteye.ipynb          # Model eğitimi ve veri analizi
├── unet_1.h5               # Eğitilmiş model dosyası
├── 111335_sat_54.jpg       # Örnek satellit görüntüsü
└── README.md               # Proje dokümantasyonu
```

## 🔧 Kurulum ve Kullanım

### Gerekli Kütüphaneler

```bash
pip install tensorflow
pip install customtkinter
pip install opencv-python
pip install matplotlib
pip install pillow
pip install numpy
pip install pandas
pip install segmentation-models
```

### Çalıştırma

1. Repoyu klonlayın:
```bash
git clone https://github.com/CorenGt/ForestEye.git
cd ForestEye
```

2. Desktop uygulamasını çalıştırın:
```bash
python app.py
```

3. "Fotoğraf Yükle" butonuna tıklayarak satellit görüntüsünü seçin
4. "Tahmin Et" butonu ile segmentasyon işlemini başlatın

## 📈 Model Performansı

Model, Kaggle'dan alınan augmented forest segmentation veri seti üzerinde eğitilmiştir:

- **Eğitim Verisi**: 4,086 görüntü
- **Doğrulama Verisi**: 511 görüntü  
- **Test Verisi**: 511 görüntü
- **Epoch**: 20
- **Batch Size**: 16

### Metrik Sonuçları
- **IoU Score**: ~0.68
- **F1-Score**: ~0.81
- **Binary Accuracy**: ~0.82

## 🎥 Ekran Görüntüleri

[Buraya uygulamanın ekran görüntülerini ekleyebilirsiniz]

## 📚 Veri Seti

Proje, Kaggle'daki [Augmented Forest Segmentation Dataset](https://www.kaggle.com/datasets/quadeer15sh/augmented-forest-segmentation) kullanılarak geliştirilmiştir.

## 🤝 Katkıda Bulunma

1. Fork'layın
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit'leyin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'i push'layın (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📧 İletişim

Herhangi bir sorunuz veya öneriniz varsa benimle iletişime geçebilirsiniz:
- LinkedIn: [linkedin.com/in/batuyılmaz]
- Email: [batuhanyilmaz0011@gmail.com]

---

⭐ Eğer bu proje size yardımcı olduysa, lütfen yıldızlamayı unutmayın! 