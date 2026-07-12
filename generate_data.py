import pandas as pd
import numpy as np

# Aynı veriyi tekrar üretebilmek için
np.random.seed(42)

# Kaç müşteri oluşturulacak?
n = 1000
# Müşteri bilgileri

customer_id = np.arange(1001, 1001 + n)

age = np.random.randint(18, 70, n)

gender = np.random.choice(
    ["Kadın", "Erkek"],
    n
)

city = np.random.choice(
    ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"],
    n
)

portfolio_value = np.random.randint(
    5000,
    1000000,
    n
)

last_login_days = np.random.randint(
    0,
    180,
    n
)

last_trade_days = np.random.randint(
    0,
    180,
    n
)

monthly_trade_count = np.random.randint(
    0,
    30,
    n
)
# Kampanyaya tıklama (0 = Hayır, 1 = Evet)
campaign_click = np.random.choice([0, 1], n)

# Şikayet sayısı
complaints = np.random.randint(0, 6, n)

# Kullanılan yatırım ürünü sayısı
product_count = np.random.randint(1, 6, n)

# Son 3 ayda para çekme tutarı (TL)
cash_out_3m = np.random.randint(0, 300000, n)
# Churn kuralları
churn = (
    (last_trade_days > 90) &
    (last_login_days > 60) &
    (monthly_trade_count < 3)
).astype(int)
df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "gender": gender,
    "city": city,
    "portfolio_value": portfolio_value,
    "last_login_days": last_login_days,
    "last_trade_days": last_trade_days,
    "monthly_trade_count": monthly_trade_count,
    "campaign_click": campaign_click,
    "complaints": complaints,
    "product_count": product_count,
    "cash_out_3m": cash_out_3m,
    "churn": churn
})
df.to_excel("data/investment_customers.xlsx", index=False)

print("Veri seti başarıyla oluşturuldu!")
print(df.head())