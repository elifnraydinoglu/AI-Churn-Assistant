import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# VERİYİ OKU
# ==========================================

df = pd.read_excel(
    "data/investment_customers_history.xlsx"
)


print(df.head())
print("\nHam veri boyutu:", df.shape)



# ==========================================
# MÜŞTERİ BAZLI ÖZET OLUŞTUR
# ==========================================

customer_summary = []


for customer_id, customer in df.groupby("customer_id"):

    customer = customer.sort_values("month")

    first3 = customer.head(3)
    last3 = customer.tail(3)


    summary = {}


    summary["customer_id"] = customer_id

    summary["age"] = customer["age"].iloc[0]

    summary["gender"] = customer["gender"].iloc[0]

    summary["city"] = customer["city"].iloc[0]


    # Portföy

    summary["portfolio_avg"] = (
        customer["portfolio_value"].mean()
    )


    summary["portfolio_last"] = (
        customer["portfolio_value"].iloc[-1]
    )


    summary["portfolio_growth"] = (

        (
            customer["portfolio_value"].iloc[-1]
            -
            customer["portfolio_value"].iloc[0]
        )

        /

        customer["portfolio_value"].iloc[0]

    )



    # İşlem

    summary["avg_trade"] = (
        customer["monthly_trade_count"].mean()
    )


    summary["trade_decline"] = (

        first3["monthly_trade_count"].mean()
        -
        last3["monthly_trade_count"].mean()

    ) / max(first3["monthly_trade_count"].mean(),1)



    # Login

    summary["avg_login"] = (
        customer["login_count"].mean()
    )


    summary["login_decline"] = (

        first3["login_count"].mean()
        -
        last3["login_count"].mean()

    ) / max(first3["login_count"].mean(),1)



    # Para hareketleri

    summary["cash_in_total"] = (
        customer["cash_in"].sum()
    )


    summary["cash_out_total"] = (
        customer["cash_out"].sum()
    )



    # Kampanya

    summary["campaign_rate"] = (
        customer["campaign_click"].mean()
    )


    # Şikayet

    summary["complaints"] = (
        customer["complaints"].sum()
    )


    # Pasif ay

    summary["inactive_months"] = (

        customer["monthly_trade_count"] == 0

    ).sum()



    # ======================================
    # RİSK SKORU
    # ======================================

    risk_score = 0


    if summary["trade_decline"] > 0.60:
        risk_score += 25


    if summary["login_decline"] > 0.50:
        risk_score += 20


    if summary["portfolio_growth"] < -0.25:
        risk_score += 20


    if summary["cash_out_total"] > summary["cash_in_total"]:
        risk_score += 15


    if summary["campaign_rate"] < 0.30:
        risk_score += 10


    if summary["complaints"] >= 5:
        risk_score += 10


    if summary["inactive_months"] >= 3:
        risk_score += 15



    summary["risk_score"] = min(
        risk_score,
        100
    )


    summary["churn"] = (

        1 if risk_score >= 50 else 0

    )


    customer_summary.append(summary)



# ==========================================
# DATAFRAME
# ==========================================

customer_summary = pd.DataFrame(
    customer_summary
)


print(
    "\nMüşteri sayısı:",
    len(customer_summary)
)



# ==========================================
# KATEGORİK DÖNÜŞÜM
# ==========================================

customer_summary = pd.get_dummies(

    customer_summary,

    columns=[
        "gender",
        "city"
    ],

    drop_first=True

)



# ==========================================
# MODEL VERİSİ
# ==========================================

X = customer_summary.drop(

    columns=[
        "customer_id",
        "churn"
    ]

)


y = customer_summary["churn"]



feature_columns = X.columns.tolist()


joblib.dump(
    feature_columns,
    "feature_columns.pkl"
)



# ==========================================
# TRAIN TEST
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)



# ==========================================
# RANDOM FOREST
# ==========================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=10,

    random_state=42,

    class_weight="balanced"

)



model.fit(

    X_train,

    y_train

)



# ==========================================
# TEST
# ==========================================

prediction = model.predict(
    X_test
)


print(
    "\nAccuracy:",
    round(
        accuracy_score(
            y_test,
            prediction
        ),
        4
    )
)


print(
    classification_report(
        y_test,
        prediction
    )
)



# ==========================================
# KAYDET
# ==========================================

joblib.dump(

    model,

    "churn_model.pkl"

)


customer_summary.to_excel(

    "customer_summary.xlsx",

    index=False

)


print("\n✅ churn_model.pkl oluşturuldu")
print("✅ feature_columns.pkl oluşturuldu")
print("✅ customer_summary.xlsx oluşturuldu")