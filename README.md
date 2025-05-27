# 📊 Facebook Gönderilerinin Beğeni Sayısına Göre Sınıflandırılması

## 📌 Proje Özeti
Bu projede, Facebook gönderilerinin beğeni (like) sayısına göre sınıflandırılması hedeflenmiştir. Karar Ağacı (Decision Tree) algoritması kullanılarak gönderilerin "Düşük", "Orta" ve "Yüksek" beğeni kategorilerine ayrılması amaçlanmıştır.

## 🧠 Kullanılan Yöntem
- **Model:** Decision Tree (CART – Gini Index)
- **Kütüphaneler:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Veri Seti:** [Facebook Metrics Dataset - Moro et al., 2016](https://doi.org/10.1016/j.jbusres.2016.02.010)
- **Doğruluk:** %95.96

## 📁 Veri Seti
Facebook gönderilerine ait 500’den fazla kayıt içerir. Öznitelikler:
- Gönderi tipi
- Yayın zamanı (hafta içi, saat aralığı)
- Sayfa toplam takipçi sayısı
- Hedef değişken: `like` (sınıflandırılmış olarak)

## 🗂️ Dosya Yapısı

```
📦proje_klasoru
 ┣ 📜proje.py
 ┣ 📊dataset_Facebook.csv
 ┣ 📈conf_matrix.png
 ┣ 🌳decision_tree.png
```

## ▶️ Çalıştırma Talimatları

1. Ortamı oluştur:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Python dosyasını çalıştır:

```bash
python proje.py
```

> Bu komut: 
> - Veriyi işler
> - Karar ağacı modelini eğitir
> - Sonuçları terminale yazdırır
> - Confusion matrix ve karar ağacını `.png` olarak kaydeder

## 📌 Gereksinimler (`requirements.txt`)
```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 📚 Kaynak
Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. *Journal of Business Research*.  
DOI: [10.1016/j.jbusres.2016.02.010](https://doi.org/10.1016/j.jbusres.2016.02.010)
