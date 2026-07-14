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
# MODEL DOSYALARI
# ==========================================

try:

    model = joblib.load("churn_model.pkl")

    feature_columns = joblib.load("feature_columns.pkl")

except Exception as e:

    st.error("❌ Model dosyaları yüklenemedi.")

    st.write(e)

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

st.sidebar.divider()


# ==========================================
# EXCEL YÜKLEME
# ==========================================

uploaded_file = st.sidebar.file_uploader(

    "📁 Upload Customer History Excel",

    type=["xlsx"]

)

if uploaded_file is None:

    st.info("👈 Lütfen sol menüden müşteri Excel dosyasını yükleyin.")

    st.stop()


# ==========================================
# EXCEL OKUMA
# ==========================================

try:

    history = pd.read_excel(uploaded_file)

except Exception as e:

    st.error("❌ Excel dosyası okunamadı.")

    st.write(e)

    st.stop()


# ==========================================
# GEREKLİ SÜTUNLAR
# ==========================================

required_columns = [

    "customer_id",
    "month",
    "age",
    "gender",
    "city",

    "portfolio_value",

    "monthly_return",

    "monthly_trade_count",

    "trade_volume",

    "login_count",

    "mobile_login",

    "web_login",

    "last_login_days",

    "last_trade_days",

    "cash_in",

    "cash_out",

    "campaign_click",

    "complaints",

    "fund_count",

    "stock_count",

    "product_count",

    "gold_ratio",

    "fx_ratio",

    "advisor_meeting"

]

missing_columns = [

    col for col in required_columns

    if col not in history.columns

]

if missing_columns:

    st.error("❌ Yüklenen Excel formatı uygun değil.")

    st.write("Eksik sütunlar:")

    st.write(missing_columns)

    st.stop()


st.sidebar.success("✅ Excel başarıyla doğrulandı.")

# ==========================================
# CUSTOMER SUMMARY OLUŞTURMA
# ==========================================

customer_summary_list = []

for customer_id, customer in history.groupby("customer_id"):

    customer = customer.sort_values("month")

    first3 = customer.head(3)
    last3 = customer.tail(3)

    summary = {}

    # ======================================
    # TEMEL BİLGİLER
    # ======================================

    summary["customer_id"] = customer_id
    summary["age"] = customer["age"].iloc[0]
    summary["gender"] = customer["gender"].iloc[0]
    summary["city"] = customer["city"].iloc[0]

    # ======================================
    # PORTFÖY
    # ======================================

    summary["portfolio_avg"] = customer["portfolio_value"].mean()

    summary["portfolio_last"] = customer["portfolio_value"].iloc[-1]

    summary["portfolio_growth"] = (

        customer["portfolio_value"].iloc[-1]
        -
        customer["portfolio_value"].iloc[0]

    ) / max(customer["portfolio_value"].iloc[0], 1)

    # ======================================
    # TRADE
    # ======================================

    summary["avg_trade"] = customer["monthly_trade_count"].mean()

    summary["trade_decline"] = (

        first3["monthly_trade_count"].mean()
        -
        last3["monthly_trade_count"].mean()

    ) / max(first3["monthly_trade_count"].mean(), 1)

    # ======================================
    # LOGIN
    # ======================================

    summary["avg_login"] = customer["login_count"].mean()

    summary["login_decline"] = (

        first3["login_count"].mean()
        -
        last3["login_count"].mean()

    ) / max(first3["login_count"].mean(), 1)

    # ======================================
    # CASH FLOW
    # ======================================

    summary["cash_in_total"] = customer["cash_in"].sum()

    summary["cash_out_total"] = customer["cash_out"].sum()

    # ======================================
    # DİĞER FEATURE'LAR
    # ======================================

    summary["campaign_rate"] = customer["campaign_click"].mean()

    summary["complaints"] = customer["complaints"].sum()

    summary["inactive_months"] = (
        customer["monthly_trade_count"] == 0
    ).sum()

    summary["fund_count"] = customer["fund_count"].iloc[-1]

    summary["stock_count"] = customer["stock_count"].iloc[-1]

    summary["product_count"] = customer["product_count"].iloc[-1]

    summary["gold_ratio"] = customer["gold_ratio"].iloc[-1]

    summary["fx_ratio"] = customer["fx_ratio"].iloc[-1]

    summary["advisor_meeting"] = customer["advisor_meeting"].sum()

    # ======================================
    # RISK SCORE
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

    summary["risk_score"] = min(risk_score, 100)

    customer_summary_list.append(summary)

# ==========================================
# DATAFRAME
# ==========================================

customer_summary = pd.DataFrame(customer_summary_list)

# ==========================================
# MODEL İÇİN HAZIRLIK
# ==========================================

model_data = customer_summary.copy()

for col in feature_columns:

    if col not in model_data.columns:
        model_data[col] = 0

model_data = model_data[feature_columns]

# ==========================================
# AI MODEL TAHMİNİ
# ==========================================

prediction = model.predict(model_data)

probability = model.predict_proba(model_data)[:, 1]

customer_summary["Prediction"] = np.where(
    prediction == 1,
    "Churn",
    "Active"
)

customer_summary["Churn Probability"] = (
    probability * 100
).round(2)

st.sidebar.success(
    f"✅ {len(customer_summary)} müşteri analiz edildi."
)
# ==========================================
# DASHBOARD
# ==========================================

if page == "🏠 Dashboard":

    st.title("🏠 Dashboard")

    total_customers = len(customer_summary)

    high_risk = len(
        customer_summary[
            customer_summary["Prediction"] == "Churn"
        ]
    )

    avg_risk = round(
        customer_summary["risk_score"].mean(),
        1
    )

    churn_rate = round(
        (
            customer_summary["Prediction"] == "Churn"
        ).mean() * 100,
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
            f"{avg_risk}/100"
        )

    with c4:

        st.metric(
            "🤖 Churn Rate",
            f"%{churn_rate}"
        )

    with c5:

        st.metric(
            "💰 Portfolio",
            f"₺{total_portfolio/1_000_000:,.1f} M"
        )

    st.divider()

    # ==========================================
    # GAUGE
    # ==========================================

    left, right = st.columns([1.2,1])

    with left:

        st.subheader("🎯 Average Risk Score")

        gauge = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=avg_risk,

                number={"suffix":"/100"},

                gauge={

                    "axis":{
                        "range":[0,100]
                    },

                    "bar":{
                        "color":"#007AFF"
                    },

                    "steps":[

                        {
                            "range":[0,35],
                            "color":"#34C759"
                        },

                        {
                            "range":[35,65],
                            "color":"#FFCC00"
                        },

                        {
                            "range":[65,100],
                            "color":"#FF3B30"
                        }

                    ]

                }

            )

        )

        gauge.update_layout(
            height=380
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

    # ==========================================
    # RISK SEGMENTS
    # ==========================================

    with right:

        st.subheader("👥 Risk Segments")

        segment = customer_summary.copy()

        segment["Risk Level"] = pd.cut(

            segment["risk_score"],

            bins=[0,35,65,100],

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
            height=380
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ==========================================
    # PORTFOLIO HISTOGRAM
    # ==========================================

    st.subheader("💰 Portfolio Distribution")

    fig = px.histogram(

        customer_summary,

        x="portfolio_last",

        nbins=40

    )

    fig.update_layout(

        height=420,

        xaxis_title="Portfolio",

        yaxis_title="Customer Count"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.subheader("📊 AI Feature Importance")

    importance = pd.DataFrame({

        "Feature":feature_columns,

        "Importance":model.feature_importances_

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

        xaxis_title="Importance"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # TOP RISK CUSTOMERS
    # ==========================================

    st.subheader("🚨 Top 10 Highest Risk Customers")

    st.dataframe(

        customer_summary.sort_values(

            by="Churn Probability",

            ascending=False

        )[

            [

                "customer_id",

                "risk_score",

                "Churn Probability",

                "Prediction",

                "portfolio_last"

            ]

        ].head(10),

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ==========================================
    # SAFEST CUSTOMERS
    # ==========================================

    st.subheader("🟢 Top 10 Lowest Risk Customers")

    st.dataframe(

        customer_summary.sort_values(

            by="Churn Probability",

            ascending=True

        )[

            [

                "customer_id",

                "risk_score",

                "Churn Probability",

                "Prediction",

                "portfolio_last"

            ]

        ].head(10),

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
    ].sort_values("month")

    st.divider()

    # ==========================================
    # KPI
    # ==========================================

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Risk Score",
            f'{customer["risk_score"]:.0f}/100'
        )

    with c2:
        st.metric(
            "AI Prediction",
            customer["Prediction"]
        )

    with c3:
        st.metric(
            "Probability",
            f'{customer["Churn Probability"]:.1f}%'
        )

    with c4:
        st.metric(
            "Portfolio",
            f'₺{customer["portfolio_last"]:,.0f}'
        )

    with c5:
        st.metric(
            "Average Trade",
            f'{customer["avg_trade"]:.1f}'
        )

    st.divider()

    # ==========================================
    # CUSTOMER INFO
    # ==========================================

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
                customer["gender"],
                customer["city"]
            ]

        })

        st.dataframe(
            info,
            hide_index=True,
            use_container_width=True
        )

    with right:

        st.subheader("Behavior Summary")

        st.write(f"""
**Average Login:** {customer["avg_login"]:.1f}

**Trade Decline:** %{customer["trade_decline"]*100:.1f}

**Login Decline:** %{customer["login_decline"]*100:.1f}

**Campaign Rate:** %{customer["campaign_rate"]*100:.1f}

**Complaints:** {customer["complaints"]}

**Inactive Months:** {customer["inactive_months"]}
""")

    st.divider()

    # ==========================================
    # PORTFOLIO TREND
    # ==========================================

    st.subheader("💰 Portfolio Trend")

    fig = px.line(

        history_customer,

        x="month",

        y="portfolio_value",

        markers=True

    )

    fig.update_layout(
        height=420
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # TRADE & LOGIN
    # ==========================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📈 Trade Trend")

        fig = px.line(

            history_customer,

            x="month",

            y="monthly_trade_count",

            markers=True

        )

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("📱 Login Trend")

        fig = px.line(

            history_customer,

            x="month",

            y="login_count",

            markers=True

        )

        fig.update_layout(height=350)

        st.plotly_chart(
            fig,
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

    fig.update_layout(height=420)

    st.plotly_chart(
        fig,
    use_container_width=True
    )

    st.divider()

    # ==========================================
    # AI ANALYSIS
    # ==========================================

    st.subheader("🤖 AI Customer Evaluation")

    if customer["Prediction"] == "Churn":

        st.error(f"""

### High Churn Risk Detected

**AI Prediction Probability:** %{customer["Churn Probability"]:.1f}

This customer exhibits strong churn signals.

Potential reasons include:

- Decreasing trading activity
- Lower digital engagement
- Portfolio contraction
- Cash outflows
- Weak campaign response

Immediate retention actions are recommended.

""")

    else:

        st.success(f"""

### Customer Appears Healthy

**AI Prediction Probability:** %{customer["Churn Probability"]:.1f}

The customer's investment behavior remains stable.

Continue monitoring and maintain engagement.

""")

    st.divider()

    # ==========================================
    # RECOMMENDED ACTIONS
    # ==========================================

    st.subheader("🎯 Recommended Actions")

    if customer["Prediction"] == "Churn":

        actions = [

            "📞 Relationship manager should contact the customer.",

            "💰 Offer personalized investment opportunities.",

            "🎁 Provide commission discounts or campaign incentives.",

            "📈 Recommend portfolio review with an advisor."

        ]

    else:

        actions = [

            "✅ Continue regular communication.",

            "📧 Send personalized investment newsletters.",

            "📱 Offer new investment products."

        ]

    for action in actions:
     st.write(action)

        # ==========================================
# CUSTOMER TRENDS
# ==========================================

elif page == "📈 Customer Trends":

    st.title("📈 Customer Trends")

    monthly = history.copy()

    monthly["month"] = pd.to_datetime(monthly["month"])

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

            yaxis_title="Average Portfolio"

        )

        st.plotly_chart(
         fig,
         use_container_width=True
        )

    with col2:

        st.subheader("📉 Average Monthly Trade Count")

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

        st.subheader("💸 Monthly Cash Flow")

        cash = trend[
            ["month", "cash_in", "cash_out"]
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

            "Average Monthly Trade",

            "Average Login",

            "Total Cash In",

            "Total Cash Out",

            "Average Churn Probability"

        ],

        "Value": [

            f"₺{trend['portfolio_value'].mean():,.0f}",

            round(trend["monthly_trade_count"].mean(), 2),

            round(trend["login_count"].mean(), 2),

            f"₺{trend['cash_in'].sum():,.0f}",

            f"₺{trend['cash_out'].sum():,.0f}",

            f"%{customer_summary['Churn Probability'].mean():.1f}"

        ]

    })

    st.dataframe(

      summary,

        use_container_width=True,

        hide_index=True

     )

    st.divider()

    # ==========================================
    # AI TREND SUMMARY
    # ==========================================

    st.subheader("🤖 AI Trend Summary")

    st.info(f"""

### Overall Customer Behavior

- Total Customers: **{len(customer_summary):,}**

- Average Risk Score: **{customer_summary['risk_score'].mean():.1f}/100**

- Average Churn Probability: **%{customer_summary['Churn Probability'].mean():.1f}**

- High Risk Customers: **{len(customer_summary[customer_summary['Prediction']=='Churn'])}**

### AI Observation

Customer behavior trends indicate changes in investment activity, login frequency and portfolio balances over time. These trends help identify customers who may require proactive engagement before churn occurs.

""")
    # ==========================================
# AI INSIGHTS
# ==========================================

elif page == "🤖 AI Insights":

    st.title("🤖 AI Insights")

    st.markdown(
        "Artificial Intelligence based customer churn analysis and executive insights."
    )

    st.divider()

    # ==========================================
    # FEATURE IMPORTANCE
    # ==========================================

    st.subheader("📊 Top 10 Most Important Features")

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

            text_auto=".3f"

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

    except:

        st.warning("Feature importance is not available for this model.")

    st.divider()

    # ==========================================
    # RISK DISTRIBUTION
    # ==========================================

    st.subheader("🚨 Customer Risk Distribution")

    risk_df = customer_summary.copy()

    risk_df["Risk Level"] = pd.cut(

        risk_df["risk_score"],

        bins=[0,35,65,100],

        labels=[

            "Low",

            "Medium",

            "High"

        ]

    )

    fig = px.histogram(

        risk_df,

        x="Risk Level",

        color="Risk Level",

        text_auto=True

    )

    fig.update_layout(

        height=400,

        showlegend=False

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # CITY RISK ANALYSIS
    # ==========================================

    st.subheader("🏙 Highest Risk Cities")

    city_risk = customer_summary.groupby(

        "city"

    ).agg({

        "risk_score":"mean",

        "Churn Probability":"mean",

        "customer_id":"count"

    }).reset_index()

    city_risk.columns = [

        "City",

        "Average Risk",

        "Average Probability",

        "Customer Count"

    ]

    city_risk = city_risk.sort_values(

        by="Average Risk",

        ascending=False

    )

    fig = px.bar(

        city_risk,

        x="City",

        y="Average Risk",

        text_auto=".1f"

    )

    fig.update_layout(

        height=420

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.dataframe(

        city_risk,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ==========================================
    # AI EXECUTIVE SUMMARY
    # ==========================================

    st.subheader("🧠 Executive Summary")

    high_risk = len(

        customer_summary[
            customer_summary["Prediction"]=="Churn"
        ]

    )

    avg_probability = customer_summary[
        "Churn Probability"
    ].mean()

    avg_risk = customer_summary[
        "risk_score"
    ].mean()

    avg_portfolio = customer_summary[
        "portfolio_last"
    ].mean()

    st.info(f"""

### Executive Summary

**Total Customers**

{len(customer_summary):,}

**High Risk Customers**

{high_risk:,}

**Average Risk Score**

{avg_risk:.1f}/100

**Average Churn Probability**

%{avg_probability:.1f}

**Average Portfolio**

₺{avg_portfolio:,.0f}

---

### AI Findings

• Trading activity is one of the strongest indicators of churn.

• Customers with declining login frequency show significantly higher churn probability.

• Portfolio contraction and increased cash outflows are associated with elevated risk.

• Low campaign engagement contributes to churn likelihood.

---

### Recommendation

Prioritize customers with **Churn Probability above 70%** for proactive retention campaigns, personalized investment offers, and direct relationship manager outreach.

""")

    st.divider()

    # ==========================================
    # AI PREDICTION SUMMARY
    # ==========================================

    st.subheader("📋 Prediction Summary")

    prediction_summary = customer_summary.groupby(

        "Prediction"

    ).agg({

        "customer_id":"count",

        "Churn Probability":"mean",

        "portfolio_last":"mean"

    }).reset_index()

    prediction_summary.columns = [

        "Prediction",

        "Customer Count",

        "Average Probability",

        "Average Portfolio"

    ]

    st.dataframe(

        prediction_summary,

        use_container_width=True,

        hide_index=True

    )
    # ==========================================
# PORTFOLIO ANALYTICS
# ==========================================

elif page == "📊 Portfolio Analytics":

    st.title("📊 Portfolio Analytics")

    st.markdown(
        "Portfolio, demographic and regional customer analytics."
    )

    st.divider()

    # ==========================================
    # CITY ANALYSIS
    # ==========================================

    city_analysis = customer_summary.groupby("city").agg({

        "portfolio_last":"mean",

        "risk_score":"mean",

        "Churn Probability":"mean",

        "customer_id":"count"

    }).reset_index()

    city_analysis.columns = [

        "City",

        "Average Portfolio",

        "Average Risk",

        "Average Probability",

        "Customer Count"

    ]

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🏙 Average Portfolio by City")

        fig = px.bar(

            city_analysis.sort_values(

                "Average Portfolio",

                ascending=False

            ),

            x="City",

            y="Average Portfolio",

            text_auto=".0f"

        )

        fig.update_layout(height=420)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("🚨 Average Risk by City")

        fig = px.bar(

            city_analysis.sort_values(

                "Average Risk",

                ascending=False

            ),

            x="City",

            y="Average Risk",

            text_auto=".1f"

        )

        fig.update_layout(height=420)

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

        bins=[18,30,40,50,60,100],

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

        "risk_score":"mean",

        "Churn Probability":"mean"

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

        fig.update_layout(height=400)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("⚠ Risk Score by Age Group")

        fig = px.line(

            age_analysis,

            x="Age Group",

            y="risk_score",

            markers=True

        )

        fig.update_layout(height=400)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ==========================================
    # GENDER ANALYSIS
    # ==========================================

    gender_analysis = customer_summary.groupby(

        "gender"

    ).agg({

        "portfolio_last":"mean",

        "risk_score":"mean",

        "Churn Probability":"mean",

        "customer_id":"count"

    }).reset_index()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("👥 Portfolio Distribution")

        fig = px.pie(

            gender_analysis,

            names="gender",

            values="portfolio_last",

            hole=0.60

        )

        fig.update_layout(height=380)

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("⚠ Average Risk by Gender")

        fig = px.bar(

            gender_analysis,

            x="gender",

            y="risk_score",

            text_auto=".1f"

        )

        fig.update_layout(height=380)

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

            "Churn Probability",

            "Prediction"

        ]

    ].head(20)

    st.dataframe(

        top_portfolio,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ==========================================
    # PORTFOLIO vs RISK
    # ==========================================

    st.subheader("📈 Portfolio vs Risk Score")

    fig = px.scatter(

        customer_summary,

        x="portfolio_last",

        y="risk_score",

        color="Prediction",

        size="Churn Probability",

        hover_data=[

            "customer_id",

            "city"

        ]

    )

    fig.update_layout(

        height=500,

        xaxis_title="Portfolio",

        yaxis_title="Risk Score"

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
    # ==========================================
# CUSTOMER SEGMENTATION
# ==========================================

elif page == "👥 Customer Segmentation":

    st.title("👥 Customer Segmentation")

    segmentation = customer_summary.copy()

    # ==========================================
    # SEGMENT OLUŞTUR
    # ==========================================

    segmentation["Segment"] = np.select(

        [

            segmentation["Churn Probability"] < 20,

            (segmentation["Churn Probability"] >= 20) &
            (segmentation["Churn Probability"] < 50),

            (segmentation["Churn Probability"] >= 50) &
            (segmentation["Churn Probability"] < 80),

            segmentation["Churn Probability"] >= 80

        ],

        [

            "Champions",

            "Loyal",

            "At Risk",

            "Critical"

        ],

        default="Unknown"

    )

    # ==========================================
    # SEGMENT PIE
    # ==========================================

    st.subheader("📊 Customer Segments")

    fig = px.pie(

        segmentation,

        names="Segment",

        hole=0.65

    )

    fig.update_layout(height=420)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # SEGMENT TABLE
    # ==========================================

    segment_table = segmentation.groupby(

        "Segment",

        observed=False

    ).agg({

        "customer_id":"count",

        "portfolio_last":"mean",

        "risk_score":"mean",

        "Churn Probability":"mean"

    }).reset_index()

    segment_table.columns = [

        "Segment",

        "Customer Count",

        "Average Portfolio",

        "Average Risk",

        "Average Churn Probability"

    ]

    st.subheader("📋 Segment Summary")

    st.dataframe(

        segment_table,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ==========================================
    # SEGMENT BAR
    # ==========================================

    st.subheader("📈 Average Churn Probability")

    fig = px.bar(

        segment_table,

        x="Segment",

        y="Average Churn Probability",

        text_auto=".1f"

    )

    fig.update_layout(height=400)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # SEGMENT SCATTER
    # ==========================================

    st.subheader("🎯 Portfolio vs Churn Probability")

    fig = px.scatter(

        segmentation,

        x="portfolio_last",

        y="Churn Probability",

        color="Segment",

        size="risk_score",

        hover_data=[

            "customer_id",

            "city",

            "Prediction"

        ]

    )

    fig.update_layout(

        height=500,

        xaxis_title="Portfolio Value",

        yaxis_title="Churn Probability (%)"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ==========================================
    # SEGMENT AI SUMMARY
    # ==========================================

    st.subheader("🤖 AI Segment Evaluation")

    champions = len(segmentation[segmentation["Segment"] == "Champions"])
    loyal = len(segmentation[segmentation["Segment"] == "Loyal"])
    atrisk = len(segmentation[segmentation["Segment"] == "At Risk"])
    critical = len(segmentation[segmentation["Segment"] == "Critical"])

    st.info(f"""

### AI Customer Segmentation Summary

🏆 **Champions:** {champions}

These customers have the lowest churn probability and represent the most valuable customer base.

---

💙 **Loyal:** {loyal}

Stable customers who should continue receiving personalized investment opportunities.

---

⚠ **At Risk:** {atrisk}

These customers show early warning signals and should receive targeted retention campaigns.

---

🚨 **Critical:** {critical}

Immediate action is recommended. These customers have the highest probability of churn and should be contacted proactively.

""")

    st.divider()

    # ==========================================
    # CUSTOMER LIST
    # ==========================================

    st.subheader("📄 Customer List by Segment")

    selected_segment = st.selectbox(

        "Select Segment",

        segmentation["Segment"].unique()

    )

    filtered = segmentation[

        segmentation["Segment"] == selected_segment

    ]

    st.dataframe(

        filtered[

            [

                "customer_id",

                "city",

                "portfolio_last",

                "risk_score",

                "Churn Probability",

                "Prediction"

            ]

        ],

        use_container_width=True,

        hide_index=True

    )
    # ==========================================
# BATCH PREDICTION
# ==========================================

elif page == "📁 Batch Prediction":

    st.title("📁 Batch Prediction")

    st.success(
        "✅ Uploaded customer history has been analyzed successfully."
    )

    st.write(
        f"Total Customers : {len(customer_summary):,}"
    )

    st.divider()

    # ==========================================
    # PREDICTION RESULTS
    # ==========================================

    st.subheader("🤖 AI Prediction Results")

    result = customer_summary.copy()

    result = result.sort_values(

        by="Churn Probability",

        ascending=False

    )

    st.dataframe(

        result,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # ==========================================
    # SUMMARY
    # ==========================================

    total = len(result)

    churn = len(

        result[
            result["Prediction"] == "Churn"
        ]

    )

    active = len(

        result[
            result["Prediction"] == "Active"
        ]

    )

    avg_probability = result[
        "Churn Probability"
    ].mean()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Customers",
            total
        )

    with c2:

        st.metric(
            "Predicted Churn",
            churn
        )

    with c3:

        st.metric(
            "Average Probability",
            f"{avg_probability:.1f}%"
        )

    st.divider()

    # ==========================================
    # PREDICTION CHART
    # ==========================================

    st.subheader("Prediction Distribution")

    fig = px.histogram(

        result,

        x="Churn Probability",

        nbins=20,

        color="Prediction"

    )

    fig.update_layout(

        height=420

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

    st.divider()

    # ==========================================
    # DOWNLOAD CSV
    # ==========================================

    csv = result.to_csv(

        index=False

    ).encode("utf-8")

    st.download_button(

        "📥 Download CSV",

        csv,

        "prediction_results.csv",

        "text/csv"

    )

    # ==========================================
    # DOWNLOAD EXCEL
    # ==========================================

    import io

    output = io.BytesIO()

    with pd.ExcelWriter(

        output,

        engine="openpyxl"

    ) as writer:

        result.to_excel(

            writer,

            index=False,

            sheet_name="Prediction Results"

        )

    st.download_button(

        "📥 Download Excel",

        output.getvalue(),

        "prediction_results.xlsx",

        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    )

    st.divider()

    # ==========================================
    # AI SUMMARY
    # ==========================================

    st.subheader("🧠 AI Executive Summary")

    st.info(f"""

### Prediction Completed Successfully

- Total Customers: **{total:,}**

- Predicted Churn Customers: **{churn:,}**

- Active Customers: **{active:,}**

- Average Churn Probability: **%{avg_probability:.1f}**

### AI Recommendation

Customers with the highest churn probability should be prioritized for proactive retention campaigns, personalized investment recommendations and relationship manager engagement.

""")