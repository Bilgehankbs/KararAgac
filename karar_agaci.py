"""
Karar Ağacı (Decision Tree) Algoritması Uygulaması
Makine Öğrenmesi Dersi - Iris ve Diabetes Veri Setleri ile Karar Ağacı Uygulamaları
"""

from sklearn.datasets import load_iris, load_diabetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.inspection import DecisionBoundaryDisplay
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("1. İRİS VERİ SETİ İLE SINIFLANDIRMA PROBLEMİ")
print("=" * 60)

# Veri setini yükle
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print("\nVeri setinin ilk 7 satırı:")
print(df.head(7))

# Target sütununu ekle
df["target"] = iris.target
print("\nTarget sütunu eklenmiş veri seti:")
print(df)

# Features ve target'ı ayır
X = iris.data  # features
y = iris.target  # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı modelini oluştur ve eğit
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

# Model değerlendirmesi
y_pred = tree_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nİris veri seti ile eğitilen DT modeli doğruluğu: {accuracy}")

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Target isimleri
print(f"\nSınıf isimleri: {iris.target_names}")

# Karar ağacını görselleştir
plt.figure(figsize=(15, 10))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.title("İris Veri Seti için Karar Ağacı")
plt.show()

# Özellik önemleri
print("\nÖzellik önemleri (önemli olanlar en üstte):")
feature_importances = tree_clf.feature_importances_
feature_names = iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse=True)

for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name}: {importance}")

print("\nEn önemli özellik: petal length (cm) - 0.90 önem skoru")


print("\n" + "=" * 60)
print("2. KARAR SINIRLARINI GÖRSELLEŞTİRME")
print("=" * 60)

# Veri setini tekrar yükle (görselleştirme için)
iris = load_iris()

n_classes = len(iris.target_names)
plot_colors = "ryb"

plt.figure(figsize=(15, 10))

for pairidx, pair in enumerate([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]):
    X = iris.data[:, pair]
    y = iris.target

    clf = DecisionTreeClassifier().fit(X, y)

    ax = plt.subplot(2, 3, pairidx+1)
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    DecisionBoundaryDisplay.from_estimator(clf,
                                         X,
                                         cmap=plt.cm.RdYlBu,
                                         response_method="predict",
                                         ax=ax,
                                         xlabel=iris.feature_names[pair[0]],
                                         ylabel=iris.feature_names[pair[1]])

    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i], 
                   cmap=plt.cm.RdYlBu, edgecolors="black")

plt.legend()
plt.suptitle("İris Veri Seti - Karar Sınırları Görselleştirmesi")
plt.show()


print("\n" + "=" * 60)
print("3. DİABETES VERİ SETİ İLE REGRESYON PROBLEMİ")
print("=" * 60)

# Diabetes veri setini yükle
diabetes = load_diabetes()

X = diabetes.data  # features
y = diabetes.target  # target

print(f"Diabetes veri seti özellik sayısı: {len(diabetes.feature_names)}")
print(f"Örnek sayısı: {len(diabetes.target)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı regresyon modeli
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)

# Model değerlendirmesi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"\nMSE: {mse}")
print(f"RMSE: {rmse}")

print(f"\nDiabetes özellik isimleri: {diabetes.feature_names}")


print("\n" + "=" * 60)
print("4. YAPAY VERİ SETİ İLE REGRESYON PROBLEMİ")
print("=" * 60)

# Yapay veri seti oluştur
print("Yapay veri seti oluşturuluyor...")
X = np.sort(5 * np.random.rand(80, 1), axis=0)  # feature
y = np.sin(X).ravel()  # target
y[::5] += 5 * (0.5 - np.random.rand(16))  # noise

print("Veri seti açıklaması:")
print("- X: 0-5 arası sıralanmış 80 rastgele sayı")
print("- y: X'in sinüs değerleri + gürültü")
print("- Gürültü: Her 5. veri noktasına rastgele değer eklendi")

# İki farklı max_depth ile model oluştur
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_1.fit(X, y)

regr_2 = DecisionTreeRegressor(max_depth=15)
regr_2.fit(X, y)

# Test verisi oluştur
X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

# Sonuçları görselleştir
plt.figure(figsize=(10, 6))
plt.plot(X, y, c="red", label="Gerçek Veri", linewidth=2)
plt.scatter(X, y, c="red", label="Eğitim Verisi", s=50)
plt.plot(X_test, y_pred_1, color="blue", label="Max Depth: 2", linewidth=2)
plt.plot(X_test, y_pred_2, color="green", label="Max Depth: 15", linewidth=2)
plt.xlabel("Veri")
plt.ylabel("Hedef")
plt.title("Karar Ağacı Regresyon - Farklı Max Depth Değerleri")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nSonuç analizi:")
print("- Max Depth = 2: Daha basit model, az aşırı öğrenme")
print("- Max Depth = 15: Daha karmaşık model, aşırı öğrenme riski")




