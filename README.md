# 📊 Facebook Gönderilerinin Beğeni Sayısına Göre Sınıflandırılması


## 📌 Proje Özeti

Bu projede, Facebook gönderilerinin beğeni (like) sayısı temel alınarak sınıflandırılması amaçlanmıştır. Karar Ağacı algoritması kullanılarak gönderiler **"Düşük", "Orta" ve "Yüksek"** olmak üzere üç beğeni kategorisine ayrılmıştır. Modelin doğruluğu ve performansı çeşitli metriklerle analiz edilmiş; ROC eğrisi, karışıklık matrisi ve karar ağacı görsel olarak değerlendirilmiştir.

Proje Kod Video Linki: https://youtu.be/MhYZH_VcXEQ

---

## 🧠 Kullanılan Yöntem ve Araçlar

- 📘 **Model:** `DecisionTreeClassifier` (CART – Gini Index)
- 📚 **Kütüphaneler:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 📁 Veri Seti Bilgisi

- **Kaynak:** [Facebook Metrics Dataset](https://archive.ics.uci.edu/dataset/368/facebook+metrics) (Moro et al., 2016, DOI: [10.1016/j.jbusres.2016.02.010](https://doi.org/10.1016/j.jbusres.2016.02.010))
- **Kayıt Sayısı:** 495
- **Bazı Öznitelikler:**
  - `Page total likes`
  - `Type` (Gönderi tipi)
  - `Post Weekday`, `Post Hour`
  - `Lifetime Post Impressions`
  - `share`, `comment`, `Total Interactions`
- **Hedef Değişken:** `like` → `"Düşük"`, `"Orta"`, `"Yüksek"` olarak etiketlendi

---

## 📈 Sonuçlar

- **Doğruluk:** %95.3
- **En önemli öznitelik:** `Total Interactions`
- **ROC AUC Değerleri:**
  - Düşük: 0.97
  - Orta: 0.97
  - Yüksek: 1.00
  - Mikro Ortalama: 0.97

---

## 🗂️ Dosya Yapısı

```
📦 facebook_like_classification
 ┣ 📜 proje.py
 ┣ 📊 dataset_Facebook.csv
 ┣ 📈 confusion_matrix.png
 ┣ 🌳 decision_tree.png
 ┣ 🔍 ozellik_onemlilikleri.png
 ┣ 📉 roc_egri_cok_sinifli.png
 ┣ 📄 README.md

```

---

## ▶️ Çalıştırma Talimatları

1. Ortamı oluştur ve bağımlılıkları yükle:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Ana Python dosyasını çalıştır:

```bash
python proje.py
```

> Bu komut:
> - Veriyi işler ve kategorilere ayırır  
> - Karar ağacı modelini eğitir  
> - ROC eğrisi ve metrikleri hesaplar  
> - Aşağıdaki görselleri üretir:
>   - `confusion_matrix.png`
>   - `decision_tree.png`
>   - `ozellik_onemlilikleri.png`
>   - `roc_egri_cok_sinifli.png`

---

## 📌 Gereksinimler (`requirements.txt`)

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

---

## 📚 Kaynakça

Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. *Journal of Business Research, 69*(9), 3341–3351. https://doi.org/10.1016/j.jbusres.2016.02.010


## 📚 Kaynak
Moro, S., Rita, P., & Vala, B. (2016). Predicting social media performance metrics and evaluation of the impact on brand building: A data mining approach. *Journal of Business Research*.  
DOI: [10.1016/j.jbusres.2016.02.010](https://doi.org/10.1016/j.jbusres.2016.02.010)
