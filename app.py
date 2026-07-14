import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

# ==========================================
# SAYFA AYARLARI
# ==========================================

st.set_page_config(
    page_title="AI Customer Intelligence Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CSS
# ==========================================

css_file = Path("style.css")

if css_file.exists():
    with open(css_file, encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

# ==========================================
# BAŞLIK
# ==========================================

st.markdown("""
# 🤖 AI Customer Intelligence Platform

### AI-Powered Customer Churn Prediction & Portfolio Analytics
""")

st.divider()

# ==========================================
# MODEL VE VERİ
# ==========================================

model = joblib.load("churn_model.pkl")

feature_columns = joblib.load("feature_columns.pkl")

customer_summary = pd.read_excel(
    "customer_summary.xlsx"
)

uploaded_file = st.sidebar.file_uploader(
    "Müşteri Verisi Yükle",
    type=["xlsx"]
)

if uploaded_file:

    history = pd.read_excel(uploaded_file)

else:
    st.warning(
        "Analiz için müşteri Excel dosyası yükleyiniz."
    )
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [

        "🏠 Dashboard",

        "👤 Customer Analysis",

        "📈 Customer Trends",

        "🤖 AI Insights",

        "📊 Portfolio Analytics",

        "👥 Customer Segmentation",

        "📁 Batch Prediction"

    ]

)
# ==========================================
# DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    total_customers = len(customer_summary)

    high_risk = len(
        customer_summary[
            customer_summary["risk_score"] >= 50
        ]
    )

    avg_risk = round(
        customer_summary["risk_score"].mean(),
        1
    )

    churn_rate = round(
        customer_summary["churn"].mean() * 100,
        1
    )

    total_portfolio = customer_summary[
        "portfolio_last"
    ].sum()

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:

        st.metric(

            "👥 Total Customers",

            f"{total_customers:,}"

        )

    with c2:

        st.metric(

            "🚨 High Risk",

            f"{high_risk:,}"

        )

    with c3:

        st.metric(

            "📊 Avg Risk Score",

            avg_risk

        )

    with c4:

        st.metric(

            "📉 Churn Rate",

            f"%{churn_rate}"

        )

    with c5:

        st.metric(

            "💰 Portfolio",

            f"₺{total_portfolio/1_000_000:,.1f} M"

        )

    st.divider()

        # ==========================================
    # DASHBOARD GRAFİKLERİ
    # ==========================================

    left, right = st.columns([1.2, 1])

    # ------------------------------------------
    # RİSK DAĞILIMI (GAUGE)
    # ------------------------------------------

    with left:

        st.subheader("🎯 Customer Risk Score")

        gauge = go.Figure(
            go.Indicator(

                mode="gauge+number",

                value=avg_risk,

                number={"suffix": "/100"},

                gauge={

                    "axis": {

                        "range": [0, 100]

                    },

                    "bar": {

                        "color": "#007AFF"

                    },

                    "steps": [

                        {

                            "range": [0, 35],

                            "color": "#34C759"

                        },

                        {

                            "range": [35, 65],

                            "color": "#FFCC00"

                        },

                        {

                            "range": [65, 100],

                            "color": "#FF3B30"

                        }

                    ]

                }

            )

        )

        gauge.update_layout(

            height=380,

            margin=dict(
                l=30,
                r=30,
                t=30,
                b=20
            )

        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    # ------------------------------------------
    # RİSK SEGMENTLERİ
    # ------------------------------------------

    with right:

        st.subheader("👥 Risk Segments")

        segment = customer_summary.copy()

        segment["Risk Level"] = pd.cut(

            segment["risk_score"],

            bins=[0, 35, 65, 100],

            labels=[

                "Low",

                "Medium",

                "High"

            ]

        )

        fig = px.pie(

            segment,

            names="Risk Level",

            hole=0.65

        )

        fig.update_layout(

            height=380,

            showlegend=True

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ==========================================
    # PORTFÖY DAĞILIMI
    # ==========================================

    st.subheader("💰 Portfolio Distribution")

    portfolio_chart = px.histogram(

        customer_summary,

        x="portfolio_last",

        nbins=40

    )

    portfolio_chart.update_layout(

        height=420,

        xaxis_title="Portfolio Value",

        yaxis_title="Customer Count"

    )

    st.plotly_chart(

        portfolio_chart,

        use_container_width=True
        )

    st.divider()

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.subheader("📊 Most Important Risk Factors")

    try:

        importance = pd.DataFrame({

            "Feature": feature_columns,

            "Importance": model.feature_importances_

        })

        importance = importance.sort_values(
            by="Importance",
            ascending=False
        ).head(10)

        fig = px.bar(

            importance,

            x="Importance",

            y="Feature",

            orientation="h",

            text_auto=".2f"

        )

        fig.update_layout(

            height=450,

            yaxis_title="",

            xaxis_title="Importance Score"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    except Exception as e:

        st.warning(
            "Feature importance görüntülenemedi."
        )

    st.divider()

    # ==========================================
    # EN RİSKLİ MÜŞTERİLER
    # ==========================================

    st.subheader("🚨 Top 10 Highest Risk Customers")

    top_risk = customer_summary.sort_values(

        by="risk_score",

        ascending=False

    )[

        [

            "customer_id",

            "risk_score",

            "portfolio_last",

            "trade_decline",

            "login_decline"

        ]

    ].head(10)

    st.dataframe(

        top_risk,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ==========================================
    # EN GÜVENLİ MÜŞTERİLER
    # ==========================================

    st.subheader("🟢 Top 10 Lowest Risk Customers")

    safest = customer_summary.sort_values(

        by="risk_score",

        ascending=True

    )[

        [

            "customer_id",

            "risk_score",

            "portfolio_last",

            "trade_decline",

            "login_decline"

        ]

    ].head(10)

    st.dataframe(

        safest,

        use_container_width=True,

        hide_index=True

    )
    # ==========================================
# CUSTOMER ANALYSIS
# ==========================================

elif page == "👤 Customer Analysis":

    st.title("👤 Customer Analysis")

    customer_list = sorted(
        customer_summary["customer_id"].unique()
    )

    selected_customer = st.selectbox(

        "Select Customer",

        customer_list

    )

    customer = customer_summary[
        customer_summary["customer_id"] == selected_customer
    ].iloc[0]

    history_customer = history[
        history["customer_id"] == selected_customer
    ].copy()

    history_customer = history_customer.sort_values("month")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Risk Score",
            f'{customer["risk_score"]:.0f}/100'
        )

    with c2:

        st.metric(
            "Portfolio",
            f'₺{customer["portfolio_last"]:,.0f}'
        )

    with c3:

        st.metric(
            "Average Trade",
            f'{customer["avg_trade"]:.1f}'
        )

    with c4:

        churn_text = "High" if customer["churn"] == 1 else "Low"

        st.metric(
            "Churn Risk",
            churn_text
        )

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("Customer Information")

        info = pd.DataFrame({

            "Information":[

                "Customer ID",

                "Age",

                "Gender",

                "City"

            ],

            "Value":[

                customer["customer_id"],

                customer["age"],

                history_customer["gender"].iloc[0],

                history_customer["city"].iloc[0]

            ]

        })

        st.dataframe(

            info,

            hide_index=True,

            use_container_width=True

        )

    with right:

        st.subheader("Behavior Summary")

        st.write(
            f"""
**Average Login :** {customer["avg_login"]:.1f}

**Trade Decline :** %{customer["trade_decline"]*100:.1f}

**Login Decline :** %{customer["login_decline"]*100:.1f}

**Campaign Response :** %{customer["campaign_rate"]*100:.1f}

**Complaints :** {customer["complaints"]}

**Inactive Months :** {customer["inactive_months"]}
"""
        )

    st.divider()

    st.subheader("📈 Portfolio Trend")

    portfolio_fig = px.line(

        history_customer,

        x="month",

        y="portfolio_value",

        markers=True

    )

    portfolio_fig.update_layout(

        height=450,

        xaxis_title="Month",

        yaxis_title="Portfolio"

    )

    st.plotly_chart(
        portfolio_fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # TRADE & LOGIN TREND
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📉 Monthly Trade Trend")

        trade_fig = px.line(
            history_customer,
            x="month",
            y="monthly_trade_count",
            markers=True
        )

        trade_fig.update_layout(
            height=380,
            xaxis_title="Month",
            yaxis_title="Trade Count"
        )

        st.plotly_chart(
            trade_fig,
            use_container_width=True
        )

    with col2:

        st.subheader("📱 Login Trend")

        login_fig = px.line(
            history_customer,
            x="month",
            y="login_count",
            markers=True
        )

        login_fig.update_layout(
            height=380,
            xaxis_title="Month",
            yaxis_title="Login Count"
        )

        st.plotly_chart(
            login_fig,
            use_container_width=True
        )
    st.divider()

    # ==========================================
    # CASH FLOW
    # ==========================================

    st.subheader("💸 Monthly Cash Flow")

    cash = history_customer[

        [

            "month",

            "cash_in",

            "cash_out"

        ]

    ].copy()

    cash = cash.melt(

        id_vars="month",

        value_vars=[

            "cash_in",

            "cash_out"

        ],

        var_name="Type",

        value_name="Amount"

    )

    cash_fig = px.bar(

        cash,

        x="month",

        y="Amount",

        color="Type",

        barmode="group"

    )

    cash_fig.update_layout(

        height=420

    )

    st.plotly_chart(

        cash_fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # AI ANALYSIS
    # ==========================================

    st.subheader("🤖 AI Customer Analysis")

    explanation = []

    if customer["trade_decline"] > 0.60:
        explanation.append(
            "• İşlem hacmi son 12 ay içerisinde ciddi seviyede azalmış."
        )

    if customer["login_decline"] > 0.50:
        explanation.append(
            "• Dijital kanallara giriş sıklığında belirgin düşüş görülüyor."
        )

    if customer["portfolio_growth"] < -0.25:
        explanation.append(
            "• Portföy değeri önemli ölçüde küçülmüş."
        )

    if customer["cash_out_total"] > customer["cash_in_total"]:
        explanation.append(
            "• Nakit çıkışları girişlerden daha yüksek."
        )

    if customer["campaign_rate"] < 0.30:
        explanation.append(
            "• Kampanyalara katılım oldukça düşük."
        )

    if customer["complaints"] >= 5:
        explanation.append(
            "• Müşteri şikayet sayısı yüksek."
        )

    if customer["inactive_months"] >= 3:
        explanation.append(
            "• Uzun süredir işlem yapılmayan aylar mevcut."
        )

    if len(explanation) == 0:

        st.success(
            "Bu müşterinin davranışlarında belirgin bir churn sinyali tespit edilmedi."
        )

    else:

        st.warning("\n".join(explanation))

    st.divider()

    # ==========================================
    # RECOMMENDED ACTIONS
    # ==========================================

    st.subheader("🎯 Recommended Actions")

    actions = []

    if customer["risk_score"] >= 70:

        actions.extend([

            "📞 Müşteri temsilcisi tarafından aranmalı",

            "💰 Özel yatırım kampanyası sunulmalı",

            "🎁 Komisyon indirimi önerilmeli",

            "👨‍💼 Portföy danışmanı atanmalı"

        ])

    elif customer["risk_score"] >= 50:

        actions.extend([

            "📧 Kişiselleştirilmiş e-posta gönder",

            "📱 Mobil uygulama bildirimi gönder",

            "📈 Yeni fon önerileri sun"

        ])

    else:

        actions.append(
            "✅ Mevcut müşteri ilişkisi sağlıklı görünüyor."
        )

    for action in actions:

        st.write(action)

    # ==========================================
# CUSTOMER TRENDS
# ==========================================

elif page == "📈 Customer Trends":

    st.title("📈 Customer Trends")

    monthly = history.copy()

    monthly["month"] = pd.to_datetime(monthly["month"])

    st.markdown("### 📊 Overall Customer Behavior")

    trend = monthly.groupby("month").agg({

        "portfolio_value": "mean",

        "monthly_trade_count": "mean",

        "login_count": "mean",

        "cash_in": "sum",

        "cash_out": "sum"

    }).reset_index()

    # ==========================================
    # PORTFOLIO & TRADE
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("💰 Average Portfolio Value")

        fig = px.line(

            trend,

            x="month",

            y="portfolio_value",

            markers=True

        )

        fig.update_layout(

            height=380,

            xaxis_title="Month",

            yaxis_title="Portfolio"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col2:

        st.subheader("📉 Average Monthly Trade")

        fig = px.line(

            trend,

            x="month",

            y="monthly_trade_count",

            markers=True

        )

        fig.update_layout(

            height=380,

            xaxis_title="Month",

            yaxis_title="Trade Count"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ==========================================
    # LOGIN & CASH FLOW
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📱 Average Login Count")

        fig = px.line(

            trend,

            x="month",

            y="login_count",

            markers=True

        )

        fig.update_layout(

            height=380,

            xaxis_title="Month",

            yaxis_title="Login Count"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col2:

        st.subheader("💸 Cash Flow")

        cash = trend[

            [

                "month",

                "cash_in",

                "cash_out"

            ]

        ].melt(

            id_vars="month",

            var_name="Type",

            value_name="Amount"

        )

        fig = px.bar(

            cash,

            x="month",

            y="Amount",

            color="Type",

            barmode="group"

        )

        fig.update_layout(

            height=380

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ==========================================
    # CUSTOMER ACTIVITY SUMMARY
    # ==========================================

    st.subheader("📋 Customer Activity Summary")

    summary = pd.DataFrame({

        "Metric": [

            "Average Portfolio",

            "Average Trade",

            "Average Login",

            "Total Cash In",

            "Total Cash Out"

        ],

        "Value": [

            f"₺{trend['portfolio_value'].mean():,.0f}",

            round(trend["monthly_trade_count"].mean(),2),

            round(trend["login_count"].mean(),2),

            f"₺{trend['cash_in'].sum():,.0f}",

            f"₺{trend['cash_out'].sum():,.0f}"

        ]

    })

    st.dataframe(

        summary,

        hide_index=True,

        use_container_width=True

    )
    # ==========================================
# AI INSIGHTS
# ==========================================

elif page == "🤖 AI Insights":

    st.title("🤖 AI Insights")

    st.markdown(
        "Yapay zeka tarafından oluşturulan genel müşteri risk analizi"
    )

    st.divider()

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.subheader("📊 Top Risk Factors")

    importance = pd.DataFrame({

        "Feature": feature_columns,

        "Importance": model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    fig = px.bar(

        importance.head(10),

        x="Importance",

        y="Feature",

        orientation="h",

        text_auto=".3f"

    )

    fig.update_layout(

        height=450,

        yaxis_title="",

        xaxis_title="Importance"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # RISK LEVEL ANALYSIS
    # ==========================================

    st.subheader("🚨 Risk Distribution")

    low = len(customer_summary[
        customer_summary["risk_score"] < 35
    ])

    medium = len(customer_summary[
        (customer_summary["risk_score"] >= 35) &
        (customer_summary["risk_score"] < 65)
    ])

    high = len(customer_summary[
        customer_summary["risk_score"] >= 65
    ])

    risk_df = pd.DataFrame({

        "Risk":[

            "Low",

            "Medium",

            "High"

        ],

        "Customers":[

            low,

            medium,

            high

        ]

    })

    fig = px.bar(

        risk_df,

        x="Risk",

        y="Customers",

        text="Customers"

    )

    fig.update_layout(

        height=400

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # HIGH RISK CITIES
    # ==========================================

    st.subheader("🏙️ Highest Risk Cities")

    city_risk = history.merge(

        customer_summary[
            [

                "customer_id",

                "risk_score"

            ]

        ],

        on="customer_id"

    )

    city_risk = city_risk.groupby(

        "city"

    )["risk_score"].mean().reset_index()

    city_risk = city_risk.sort_values(

        by="risk_score",

        ascending=False

    )

    fig = px.bar(

        city_risk.head(10),

        x="city",

        y="risk_score",

        text_auto=".1f"

    )

    fig.update_layout(

        height=420,

        xaxis_title="City",

        yaxis_title="Average Risk Score"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # AI MANAGEMENT SUMMARY
    # ==========================================

    st.subheader("🧠 AI Executive Summary")

    high_risk_pct = round(
        high / len(customer_summary) * 100,
        1
    )

    churn_pct = round(
        customer_summary["churn"].mean() * 100,
        1
    )

    avg_portfolio = customer_summary[
        "portfolio_last"
    ].mean()

    st.info(
f"""
### Executive Summary

• Total customer count: **{len(customer_summary):,}**

• High-risk customer ratio: **%{high_risk_pct}**

• Estimated churn rate: **%{churn_pct}**

• Average portfolio size: **₺{avg_portfolio:,.0f}**

### AI Evaluation

The strongest factors affecting churn are:

- Decline in trading activity
- Decrease in digital login frequency
- Reduction in portfolio value
- Increase in cash outflows
- Low campaign engagement

### Recommendation

Customers with a risk score above **70** should be prioritized for proactive retention campaigns and personalized investment offers.
"""
    )
    # ==========================================
# PORTFOLIO ANALYTICS
# ==========================================

elif page == "📊 Portfolio Analytics":

    st.title("📊 Portfolio Analytics")

    st.markdown(
        "Portfolio, risk and demographic analysis of customers."
    )

    st.divider()

    # ==========================================
    # CITY ANALYSIS
    # ==========================================

    city_df = history.groupby("city").agg({

        "portfolio_value": "mean",

        "monthly_trade_count": "mean",

        "login_count": "mean"

    }).reset_index()

    risk_city = history[["customer_id","city"]].drop_duplicates()

    risk_city = risk_city.merge(

        customer_summary[

            [

                "customer_id",

                "risk_score"

            ]

        ],

        on="customer_id"

    )

    risk_city = risk_city.groupby(

        "city"

    )["risk_score"].mean().reset_index()

    city_df = city_df.merge(

        risk_city,

        on="city"

    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏙 Average Portfolio by City")

        fig = px.bar(

            city_df.sort_values(

                "portfolio_value",

                ascending=False

            ),

            x="city",

            y="portfolio_value",

            text_auto=".0f"

        )

        fig.update_layout(

            height=420,

            xaxis_title="City",

            yaxis_title="Average Portfolio"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col2:

        st.subheader("🚨 Average Risk by City")

        fig = px.bar(

            city_df.sort_values(

                "risk_score",

                ascending=False

            ),

            x="city",

            y="risk_score",

            text_auto=".1f"

        )

        fig.update_layout(

            height=420,

            xaxis_title="City",

            yaxis_title="Risk Score"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ==========================================
    # AGE ANALYSIS
    # ==========================================

    age_df = customer_summary.copy()

    age_df["Age Group"] = pd.cut(

        age_df["age"],

        bins=[18,30,40,50,60,80],

        labels=[

            "18-30",

            "31-40",

            "41-50",

            "51-60",

            "60+"

        ]

    )

    age_analysis = age_df.groupby(

        "Age Group",

        observed=False

    ).agg({

        "portfolio_last":"mean",

        "risk_score":"mean"

    }).reset_index()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👤 Portfolio by Age Group")

        fig = px.bar(

            age_analysis,

            x="Age Group",

            y="portfolio_last",

            text_auto=".0f"

        )

        fig.update_layout(

            height=400

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col2:

        st.subheader("⚠ Risk by Age Group")

        fig = px.line(

            age_analysis,

            x="Age Group",

            y="risk_score",

            markers=True

        )

        fig.update_layout(

            height=400

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ==========================================
    # GENDER ANALYSIS
    # ==========================================

    gender_df = history[

        [

            "customer_id",

            "gender"

        ]

    ].drop_duplicates()

    gender_df = gender_df.merge(

        customer_summary[

            [

                "customer_id",

                "portfolio_last",

                "risk_score"

            ]

        ],

        on="customer_id"

    )

    gender_analysis = gender_df.groupby(

        "gender"

    ).agg({

        "portfolio_last":"mean",

        "risk_score":"mean"

    }).reset_index()

    st.subheader("👥 Gender Analysis")

    col1, col2 = st.columns(2)

    with col1:

        fig = px.pie(

            gender_analysis,

            names="gender",

            values="portfolio_last",

            hole=0.6

        )

        fig.update_layout(

            height=380

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    with col2:

        fig = px.bar(

            gender_analysis,

            x="gender",

            y="risk_score",

            text_auto=".1f"

        )

        fig.update_layout(

            height=380

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

    st.divider()

    # ==========================================
    # TOP PORTFOLIO CUSTOMERS
    # ==========================================

    st.subheader("🏆 Top 20 Customers by Portfolio")

    top_portfolio = customer_summary.sort_values(

        "portfolio_last",

        ascending=False

    )[

        [

            "customer_id",

            "portfolio_last",

            "risk_score",

            "avg_trade",

            "avg_login"

        ]

    ].head(20)

    st.dataframe(

        top_portfolio,

        use_container_width=True,

        hide_index=True

    )# ==========================================
# CUSTOMER SEGMENTATION
# ==========================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    segmentation = customer_summary.copy()

    segmentation["Segment"] = np.select(

        [

            segmentation["risk_score"] < 25,

            (segmentation["risk_score"] >= 25) &
            (segmentation["risk_score"] < 50),

            (segmentation["risk_score"] >= 50) &
            (segmentation["risk_score"] < 75),

            segmentation["risk_score"] >= 75

        ],

        [

            "Champions",

            "Loyal",

            "At Risk",

            "Critical"

        ],

        default="Unknown"

    )

    st.subheader("Customer Segments")

    fig = px.pie(

        segmentation,

        names="Segment",

        hole=0.60

    )

    fig.update_layout(height=450)

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    segment_table = segmentation.groupby(

        "Segment",

        observed=False

    ).agg({

        "customer_id":"count",

        "portfolio_last":"mean",

        "risk_score":"mean"

    }).reset_index()

    segment_table.columns = [

        "Segment",

        "Customer Count",

        "Average Portfolio",

        "Average Risk"

    ]

    st.dataframe(

        segment_table,

        use_container_width=True,

        hide_index=True

    )

# ==========================================
# BATCH PREDICTION
# ==========================================

elif page == "📁 Batch Prediction":

    st.title("📁 Batch Prediction")

    st.write(
        "Ham 12 aylık müşteri hareket datasını yükleyerek AI churn tahmini oluşturun."
    )


    uploaded = st.file_uploader(
        "Upload Customer History Excel",
        type=["xlsx"]
    )


    if uploaded is not None:


        history_batch = pd.read_excel(uploaded)


        st.success(
            "Excel başarıyla yüklendi."
        )


        st.write(
            "Ham veri boyutu:",
            history_batch.shape
        )


        # ===============================
        # MÜŞTERİ BAZLI ÖZET
        # ===============================


        customer_summary_batch = []


        for customer_id, customer in history_batch.groupby("customer_id"):


            customer = customer.sort_values("month")


            first3 = customer.head(3)

            last3 = customer.tail(3)


            summary = {}


            summary["customer_id"] = customer_id


            summary["age"] = customer["age"].iloc[0]


            summary["portfolio_avg"] = customer["portfolio_value"].mean()


            summary["portfolio_last"] = customer["portfolio_value"].iloc[-1]


            summary["portfolio_growth"] = (
                customer["portfolio_value"].iloc[-1]
                -
                customer["portfolio_value"].iloc[0]
            ) / max(customer["portfolio_value"].iloc[0],1)


            summary["avg_trade"] = customer["monthly_trade_count"].mean()


            summary["trade_decline"] = (
                first3["monthly_trade_count"].mean()
                -
                last3["monthly_trade_count"].mean()
            ) / max(first3["monthly_trade_count"].mean(),1)


            summary["avg_login"] = customer["login_count"].mean()


            summary["login_decline"] = (
                first3["login_count"].mean()
                -
                last3["login_count"].mean()
            ) / max(first3["login_count"].mean(),1)


            summary["cash_in_total"] = customer["cash_in"].sum()


            summary["cash_out_total"] = customer["cash_out"].sum()


            summary["campaign_rate"] = customer["campaign_click"].mean()


            summary["inactive_months"] = (
                customer["monthly_trade_count"] == 0
            ).sum()


            customer_summary_batch.append(summary)



        batch = pd.DataFrame(customer_summary_batch)



        # ===============================
        # MODEL FORMATI
        # ===============================


        batch_model = batch.copy()


        prediction = model.predict(
            batch_model[feature_columns]
        )


        probability = model.predict_proba(
            batch_model[feature_columns]
        )[:,1]


        batch["Prediction"] = prediction


        batch["Churn Probability %"] = (
            probability * 100
        ).round(2)



        st.subheader(
            "🤖 AI Prediction Results"
        )


        st.dataframe(
            batch,
            use_container_width=True
        )


        csv = batch.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(

            "📥 Download Results",

            csv,

            "churn_predictions.csv",

            "text/csv"

        )