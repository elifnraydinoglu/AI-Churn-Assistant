import matplotlib.pyplot as plt
import joblib
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Excel dosyasını oku
df = pd.read_excel("data/investment_customers.xlsx")

# İlk 5 satırı göster
print(df.head())

# Veri hakkında bilgi
print("\nVeri Boyutu:")
print(df.shape)

print("\nSütunlar:")
print(df.columns)

print("\nEksik Veri:")
print(df.isnull().sum())

print("\nChurn Dağılımı:")
print(df["churn"].value_counts())
# Kategorik verileri sayısala çevir
df = pd.get_dummies(df, columns=["gender", "city"], drop_first=True)
X = df.drop("churn", axis=1)
y = df["churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nModel Başarı Oranı:")
print(accuracy_score(y_test, y_pred))

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))
# Özellik önemleri
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nEn Önemli Değişkenler:")
print(feature_importance)
plt.figure(figsize=(10,6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()

plt.show()
# Models klasörü oluştur
os.makedirs("models", exist_ok=True)

# Modeli kaydet
joblib.dump(model, "models/churn_model.pkl")

print("\nModel başarıyla kaydedildi!")