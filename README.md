# Makine Öğrenmesi Dersi 2 - Karar Ağacı
Bu projede karar ağacı algoritmasını kullandım. Karar ağacı, makine öğrenmesinde çok popüler bir algoritma. Veriyi her adımda ikiye bölerek karar düğümleri oluşturuyor. Her düğümde en iyi özelliği seçip, o özelliğe göre veriyi ayırıyor. Bu sayede ağaç yapısında kararlar alarak tahmin yapıyor. Algoritmanın en güzel yanı, sonuçları görsel olarak anlaşılır şekilde sunması. Üç farklı problem çözdüm: İris sınıflandırma, diabetes regresyon ve yapay veri seti.

## 🎯 Ne Yaptım?

İris çiçek veri setini kullanarak 3 farklı çiçek türünü sınıflandırmaya çalıştım. Sonra diabetes veri setini kullanarak 10 özellik ile diabetes ilerlemesini tahmin ettim. Son olarak sinüs fonksiyonu + gürültü oluşturup iki farklı max_depth değeri ile karşılaştırma yaptım. Bunları yaparken de Pandas, NumPy, Scikit-learn, Matplotlib Python kütüphanelerini kullandım.

## 📝 Kodda Neler Yaptım?

İris sınıflandırmada veri setini yükledim, %80 eğitim %20 test olarak böldüm. DecisionTreeClassifier ile model oluşturdum. 
Diabetes regresyonda da aynı şekilde %80 eğitim %20 test böldüm, DecisionTreeRegressor ile model oluşturdum. 
Yapay veri setinde ise sinüs fonksiyonu + gürültü oluşturdum ve iki farklı max_depth değeri ile karşılaştırma yaptım.

## Sonuç
İris sınıflandırmada %100 doğruluk aldım. En önemli özellik petal length çıktı. 
Diabetes regresyonda MSE: 4976.80, RMSE: 70.55 değerlerini elde ettim. 
Yapay veri setinde max depth=2 daha basit model, max depth=15 ise aşırı öğrenme riski taşıdığını gördüm.
Max_depth parametresi çok kritik olduğunu çok büyükse aşırı öğrenme, çok küçükse eksik öğrenmeye evrildiğini gözlemledim

Karar ağacı algoritmasının en önemli parametresi max_depth çıktı. Çok büyükse aşırı öğrenme, çok küçükse eksik öğrenmeye yol açıyor. Kullandığım diğer parametreler: criterion ("gini" veya "entropy"), max_depth, random_state.
