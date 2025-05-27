import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

# Özellikleri ve hedefi ayır
X = df_cleaned.drop(['like', 'like_category'], axis=1)
y = df_cleaned['like_category']

# Kategorik verileri dönüştür
X = pd.get_dummies(X, drop_first=True)

# Eğitim ve test verisine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini oluştur ve eğit
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Tahmin yap
y_pred = clf.predict(X_test)

# Metrikleri yazdır
print("\n=== Sınıflandırma Raporu ===\n")
print(classification_report(y_test, y_pred))
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.show()

# Karar Ağacı Görselleştirme
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=clf.classes_)
plt.title("Karar Ağacı - Beğeni Sınıflandırması")
plt.tight_layout()
plt.show()
