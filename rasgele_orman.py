"""
Rasgele Orman (Random Forest) Algoritması Uygulaması
Makine Öğrenmesi Dersi - Yüz Tanıma ve Ev Fiyatı Tahmini ile Rasgele Orman Uygulamaları
"""

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# Olivetti faces veri setini yükle
oli = fetch_olivetti_faces()

# Yüz görüntülerini göster
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.imshow(oli.images[22], cmap="gray")
    plt.axis("off")

plt.show()

# Veri setini hazırla
X = oli.data
y = oli.target

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rasgele Orman sınıflandırıcısı oluştur ve eğit
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Tahmin yap
y_pred = rf_clf.predict(X_test)

# Doğruluk hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Acc: ", accuracy)


# Farklı ağaç sayıları ile performans analizi
agacsayisi = [5, 25, 50, 100, 150, 200, 250]
acclist = []
for i in agacsayisi:
    rf_clf = RandomForestClassifier(n_estimators=i, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    acclist.append(acc)

acclist

# Performans grafiğini çiz
plt.figure()
plt.plot(agacsayisi, acclist, marker="o", linestyle="-")
plt.title("Ağaç sayısına göre doğruluk değerleri")
plt.xlabel("Ağaç sayısı")
plt.ylabel("Accuracy")
plt.show()


# Regresyon problemi için gerekli kütüphaneleri import et
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# California Housing veri setini yükle
california_housing = fetch_california_housing()

# Veri setini hazırla
X = california_housing.data
y = california_housing.target

# Eğitim ve test setlerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rasgele Orman regresyon modeli oluştur ve eğit
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Tahmin yap
y_pred = rf_reg.predict(X_test)

# Hata hesapla
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

