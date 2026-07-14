import pandas as pd
import numpy as np


np.random.seed(42)


# ==========================
# AYARLAR
# ==========================

CUSTOMER_COUNT = 5000

months = pd.date_range(
    start="2025-01-01",
    periods=12,
    freq="MS"
)

cities = [
    "İstanbul",
    "Ankara",
    "İzmir",
    "Bursa",
    "Antalya",
    "Kocaeli",
    "Konya",
    "Adana"
]

genders = [
    "Kadın",
    "Erkek"
]

records = []

# ==========================
# MÜŞTERİ OLUŞTUR
# ==========================

for customer_id in range(1001, 1001 + CUSTOMER_COUNT):

    age = np.random.randint(20, 70)

    gender = np.random.choice(genders)

    city = np.random.choice(cities)

    # Başlangıç portföyü
    portfolio = np.random.randint(
        75000,
        1200000
    )

    # Başlangıç davranışları

    trade_base = np.random.randint(8, 35)

    login_base = np.random.randint(12, 30)

    cash_base = np.random.randint(
        5000,
        100000
    )

    complaint_level = np.random.randint(0, 2)

    campaign_rate = np.random.uniform(
        0.30,
        0.90
    )

    # Bu müşteri zamanla churn olacak mı?

    will_churn = np.random.choice(
        [0, 1],
        p=[0.75, 0.25]
    )
        # =====================================
    # 12 AYLIK MÜŞTERİ DAVRANIŞI
    # =====================================

    for month_index, month in enumerate(months):

        # -----------------------------
        # Churn olacak müşteriler
        # yıl ilerledikçe kötüleşiyor
        # -----------------------------

        if will_churn == 1:

            decline = month_index / 11

            trade_count = max(
                0,
                int(
                    trade_base -
                    decline * np.random.randint(6, 14) +
                    np.random.randint(-2, 3)
                )
            )

            login_count = max(
                0,
                int(
                    login_base -
                    decline * np.random.randint(8, 18) +
                    np.random.randint(-2, 3)
                )
            )

            portfolio_change = np.random.randint(
                -90000,
                15000
            )

        else:

            trade_count = max(
                0,
                trade_base +
                np.random.randint(-3, 4)
            )

            login_count = max(
                0,
                login_base +
                np.random.randint(-4, 5)
            )

            portfolio_change = np.random.randint(
                -25000,
                45000
            )

        # -----------------------------
        # Portföy güncelle
        # -----------------------------

        portfolio += portfolio_change

        portfolio = max(
            portfolio,
            10000
        )

        monthly_return = np.random.uniform(
            -0.12,
            0.15
        )

        trade_volume = (
            trade_count *
            np.random.randint(
                2000,
                25000
            )
        )
                # ===============================
        # NAKİT HAREKETLERİ
        # ===============================

        if will_churn == 1:

            cash_in = np.random.randint(
                0,
                int(cash_base * 0.60)
            )

            cash_out = np.random.randint(
                int(cash_base * 0.80),
                int(cash_base * 2.50)
            )

        else:

            cash_in = np.random.randint(
                int(cash_base * 0.60),
                int(cash_base * 2.20)
            )

            cash_out = np.random.randint(
                0,
                int(cash_base * 0.90)
            )

        # ===============================
        # YATIRIM ÜRÜNLERİ
        # ===============================

        fund_count = np.random.randint(1, 8)

        stock_count = np.random.randint(0, 15)

        product_count = fund_count + stock_count

        gold_ratio = round(
            np.random.uniform(0, 0.40),
            2
        )

        fx_ratio = round(
            np.random.uniform(0, 0.35),
            2
        )

        # ===============================
        # DİJİTAL KANAL KULLANIMI
        # ===============================

        mobile_login = max(
            0,
            int(login_count * np.random.uniform(0.60, 0.90))
        )

        web_login = max(
            0,
            login_count - mobile_login
        )

        # ===============================
        # DANIŞMAN GÖRÜŞMESİ
        # ===============================

        advisor_meeting = np.random.choice(
            [0, 1],
            p=[0.80, 0.20]
        )
                # ===============================
        # KAMPANYA DAVRANIŞI
        # ===============================

        if will_churn == 1:

            campaign_probability = max(
                0.05,
                campaign_rate - (month_index * 0.05)
            )

        else:

            campaign_probability = min(
                0.95,
                campaign_rate + np.random.uniform(-0.05, 0.05)
            )

        campaign_click = np.random.choice(
            [0, 1],
            p=[
                1 - campaign_probability,
                campaign_probability
            ]
        )

        # ===============================
        # ŞİKAYET DAVRANIŞI
        # ===============================

        if will_churn == 1:

            complaints = np.random.poisson(
                complaint_level + (month_index * 0.20)
            )

        else:

            complaints = np.random.poisson(
                complaint_level
            )

        # ===============================
        # PASİF GÜN SAYILARI
        # ===============================

        if login_count == 0:

            last_login_days = np.random.randint(
                45,
                120
            )

        else:

            last_login_days = max(
                0,
                int(
                    30 / (login_count + 1)
                )
            )

        if trade_count == 0:

            last_trade_days = np.random.randint(
                45,
                120
            )

        else:

            last_trade_days = max(
                0,
                int(
                    30 / (trade_count + 1)
                )
            )
                    # ===============================
        # AYLIK KAYDI OLUŞTUR
        # ===============================

        records.append({

            "customer_id": customer_id,

            "month": month.strftime("%Y-%m"),

            "age": age,

            "gender": gender,

            "city": city,

            "portfolio_value": int(portfolio),

            "monthly_return": round(
                monthly_return,
                4
            ),

            "monthly_trade_count": trade_count,

            "trade_volume": int(
                trade_volume
            ),

            "login_count": login_count,

            "mobile_login": mobile_login,

            "web_login": web_login,

            "last_login_days": last_login_days,

            "last_trade_days": last_trade_days,

            "cash_in": int(
                cash_in
            ),

            "cash_out": int(
                cash_out
            ),

            "campaign_click": campaign_click,

            "complaints": int(
                complaints
            ),

            "fund_count": fund_count,

            "stock_count": stock_count,

            "product_count": product_count,

            "gold_ratio": gold_ratio,

            "fx_ratio": fx_ratio,

            "advisor_meeting": advisor_meeting,

            # Şimdilik...
            "churn": will_churn

        })
        # Son 12 ay ortalama işlem sayısı
    # ======================================
# DATAFRAME OLUŞTUR
# ======================================

df = pd.DataFrame(records)

df = df.sort_values(
    by=["customer_id", "month"]
).reset_index(drop=True)

print(df.head())

print("\nToplam kayıt:", len(df))
print("Toplam müşteri:", df["customer_id"].nunique())

# ======================================
# EXCEL'E KAYDET
# ======================================

df.to_excel(
    "investment_customers_history.xlsx",
    index=False
)

print("\n✅ investment_customers_history.xlsx oluşturuldu.")
# ======================================
# MÜŞTERİ BAZLI 12 AYLIK ÖZET DATA
# ======================================


customer_df = df.groupby(
    "customer_id"
).agg(

    age=("age", "first"),

    gender=("gender", "first"),

    city=("city", "first"),


    # Son durum bilgileri

    last_portfolio=(
        "portfolio_value",
        "last"
    ),

    last_login=(
        "login_count",
        "last"
    ),

    last_trade_count=(
        "monthly_trade_count",
        "last"
    ),


    # 12 aylık ortalamalar

    avg_portfolio=(
        "portfolio_value",
        "mean"
    ),

    avg_login=(
        "login_count",
        "mean"
    ),

    avg_trade_volume=(
        "trade_volume",
        "mean"
    ),

    avg_cash_in=(
        "cash_in",
        "mean"
    ),

    avg_cash_out=(
        "cash_out",
        "mean"
    ),


    # Toplam davranışlar

    total_complaints=(
        "complaints",
        "sum"
    ),

    campaign_rate=(
        "campaign_click",
        "mean"
    ),

    advisor_usage=(
        "advisor_meeting",
        "mean"
    ),


    # Churn etiketi

    churn=(
        "churn",
        "first"
    )

).reset_index()



# ======================================
# TREND HESAPLARI
# ======================================


def calculate_trend(x):

    first = x.iloc[0]
    last = x.iloc[-1]

    if first == 0:
        return 0

    return round(
        (last-first)/first,
        4
    )



# Login değişimi

login_trend = (
    df.groupby("customer_id")
    ["login_count"]
    .apply(calculate_trend)
)


trade_trend = (
    df.groupby("customer_id")
    ["trade_volume"]
    .apply(calculate_trend)
)


portfolio_trend = (
    df.groupby("customer_id")
    ["portfolio_value"]
    .apply(calculate_trend)
)



customer_df["login_trend"] = (
    login_trend.values
)

customer_df["trade_trend"] = (
    trade_trend.values
)

customer_df["portfolio_trend"] = (
    portfolio_trend.values
)



# ======================================
# AKTİVİTE SKORU
# ======================================


customer_df["activity_score"] = (

    customer_df["avg_login"] * 0.3

    +

    (customer_df["avg_trade_volume"]/100000)
    *0.3

    +

    customer_df["campaign_rate"]*10*0.1

    +

    customer_df["advisor_usage"]*10*0.1

    -

    customer_df["total_complaints"]*0.2

)



# ======================================
# MODEL DATA KAYDET
# ======================================


customer_df.to_excel(
    "customer_churn_model_data.xlsx",
    index=False
)


print(
    "\n✅ customer_churn_model_data.xlsx oluşturuldu."
)

print(
    "Model veri boyutu:",
    customer_df.shape
)