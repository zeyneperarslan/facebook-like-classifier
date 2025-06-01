import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import label_binarize , StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc

# Türkçe karakter desteği
plt.rcParams["font.family"] = "Arial"

# Veri setini yükle
df = pd.read_csv("dataset_Facebook.csv", sep=';')

# Beğeni değerini kategorilere ayır
def categorize_likes(x):
    if x < 100:
        return "Düşük"
    elif x < 300:
        return "Orta"
    else:
        return "Yüksek"

df['like_category'] = df['like'].apply(categorize_likes)

# Eksik verileri temizle
df_cleaned = df.dropna()

# Özellik ve hedef ayrımı
X = df_cleaned.drop(['like', 'like_category'], axis=1)
y = df_cleaned['like_category']

# Kategorik değişkenleri dönüştür
X = pd.get_dummies(X, drop_first=True)

# Eğitim ve test verilerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini oluştur ve eğit
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Tahmin yap
y_pred = clf.predict(X_test)

# === Sınıflandırma Performansı ===
print("\n=== Sınıflandırma Raporu ===\n")
print(classification_report(y_test, y_pred))
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))

# === 1. Confusion Matrix ===
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# === 2. Karar Ağacı Görselleştirme ===
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.title("Karar Ağacı - Facebook Beğeni Sınıflandırması")
plt.tight_layout()
plt.savefig("karar_agaci.png", dpi=300)
plt.show()

# === 3. Özellik Önemlilikleri Grafiği ===
importances = clf.feature_importances_
feature_names = X.columns

importance_df = pd.DataFrame({
    'Özellik': feature_names,
    'Önem': importances
})

# Sıfır olmayanları filtrele
importance_df = importance_df[importance_df['Önem'] > 0]
importance_df = importance_df.sort_values(by='Önem', ascending=True)

if importance_df.empty:
    print("Grafik çizilemedi çünkü karar ağacı önemli özellik kullanmadı.")
else:
    plt.figure(figsize=(12, 6))
    plt.barh(importance_df['Özellik'], importance_df['Önem'], color='skyblue')
    plt.xlabel("Önem")
    plt.title("Özellik Önemlilikleri (Karar Ağacı - Facebook Beğeni Tahmini)")
    plt.tight_layout()
    plt.savefig("ozellik_onemlilikleri.png", dpi=300)
    plt.show()

# Y sınıflarını binarize et (One-vs-Rest)
y_bin = label_binarize(y, classes=["Düşük", "Orta", "Yüksek"])
n_classes = y_bin.shape[1]

# Modeli yeniden eğit (ROC için OvR wrapper ile)
classifier = OneVsRestClassifier(DecisionTreeClassifier(max_depth=5, random_state=42))
X_train_scaled = StandardScaler().fit_transform(X_train)
X_test_scaled = StandardScaler().fit_transform(X_test)
classifier.fit(X_train_scaled, label_binarize(y_train, classes=["Düşük", "Orta", "Yüksek"]))

# Tahmin olasılıklarını al
y_score = classifier.predict_proba(X_test_scaled)

# ROC eğrileri ve AUC
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(label_binarize(y_test, classes=["Düşük", "Orta", "Yüksek"])[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Mikro ortalama (tüm sınıflar üzerinden)
fpr["micro"], tpr["micro"], _ = roc_curve(label_binarize(y_test, classes=["Düşük", "Orta", "Yüksek"]).ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# ROC Eğrisi çizimi
plt.figure(figsize=(8, 6))
colors = ['orange', 'green', 'blue']
class_labels = ["Düşük", "Orta", "Yüksek"]

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'ROC - {class_labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot(fpr["micro"], tpr["micro"], color='darkred', linestyle='--',
         label=f'Mikro Ort. ROC (AUC = {roc_auc["micro"]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Yanlış Pozitif Oranı')
plt.ylabel('Doğru Pozitif Oranı')
plt.title('Karar Ağacı ROC Eğrileri (Çok Sınıflı)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_egri_cok_sinifli.png", dpi=300)
plt.show()
