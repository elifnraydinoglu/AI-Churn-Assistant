import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Churn Assistant",
    page_icon="🤖",
    layout="wide"
)

# ==========================
# CSS
# ==========================

with open("style.css") as f:
    st.markdown(
        f"<style>{f.read()}</style>",
        unsafe_allow_html=True
    )


# ==========================
# SIDEBAR
# ==========================

st.sidebar.image(
    "https://img.icons8.com/color/96/artificial-intelligence.png",
    width=90
)

st.sidebar.markdown("## AI Churn")

st.sidebar.caption("Customer Intelligence")

st.sidebar.divider()

st.sidebar.metric("Model Accuracy", "96.5%")

st.sidebar.metric("Algorithm", "Random Forest")

st.sidebar.metric("Version", "2.0")

st.sidebar.divider()

st.sidebar.markdown("### Navigation")

st.sidebar.write("• Dashboard")

st.sidebar.write("• Single Customer")

st.sidebar.write("• Batch Analysis")

st.sidebar.write("• Reports")

st.sidebar.divider()

st.sidebar.markdown("### Developer")

st.sidebar.write("Elifnur Aydınoğlu")

st.sidebar.caption("AI & CRM Analytics")

st.sidebar.markdown("---")

st.sidebar.write("### 👩‍💻 Developer")
st.sidebar.write("Elifnur Aydınoğlu")

# ==========================
# HEADER
# ==========================

st.markdown("""
<div style="
background:white;
padding:35px;
border-radius:22px;
border:1px solid #ECECEC;
box-shadow:0px 8px 25px rgba(0,0,0,.05);
margin-bottom:30px;
">

<h1 style="
font-size:42px;
margin-bottom:6px;
color:#111827;
font-weight:700;
">
AI Churn Assistant
</h1>

<p style="
font-size:21px;
color:#6B7280;
margin-top:0;
margin-bottom:8px;
">
Customer Intelligence Platform
</p>

<p style="
color:#9CA3AF;
font-size:16px;
line-height:1.7;
">
Machine Learning powered CRM dashboard for predicting customer churn,
identifying high-risk customers and recommending personalized retention strategies.
</p>

</div>
""", unsafe_allow_html=True)
# ==========================
# DATA
# ==========================

df = pd.read_excel("data/investment_customers.xlsx")

model = joblib.load(
    "models/churn_model.pkl"
)

# ==========================
# KPI
# ==========================

toplam_musteri=len(df)
toplam_churn=df["churn"].sum()
aktif=toplam_musteri-toplam_churn
oran=toplam_churn/toplam_musteri*100

c1,c2,c3,c4=st.columns(4)

with c1:
    st.metric(
        "Customers",
        f"{toplam_musteri:,}"
    )

with c2:
    st.metric(
        "Churn",
        f"{toplam_churn:,}"
    )

with c3:
    st.metric(
        "Active",
        f"{aktif:,}"
    )

with c4:
    st.metric(
        "Churn Rate",
        f"{oran:.1f}%"
    )

# ==========================
# QUICK ACTIONS
# ==========================

st.markdown("### 🚀 Quick Actions")

c1, c2, c3 = st.columns(3)

with c1:
    st.button("📂 Upload Customer File", use_container_width=True)

with c2:
    st.button("📊 View Dashboard", use_container_width=True)

with c3:
    excel = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Export Report",
        data=excel,
        file_name="AI_Churn_Report.csv",
        mime="text/csv",
        use_container_width=True
    )

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png",
        width=90
    )

    st.title("AI Churn")

    st.success("🟢 Model Online")

    st.metric(
        "Accuracy",
        "96.5%"
    )

    st.metric(
        "Algorithm",
        "Random Forest"
    )

    st.metric(
        "Version",
        "1.0"
    )

    st.markdown("---")

    st.write("### Developer")

    st.write("Elifnur Aydınoğlu")

    st.caption("AI CRM Dashboard")

# ==========================
# GRAFİKLER
# ==========================

st.divider()

left,right=st.columns(2)

with left:

    fig = px.pie(
        df,
        names="churn",
        title="Customer Risk Distribution",
        color="churn",
        color_discrete_map={
            0: "#2ecc71",
            1: "#e74c3c"
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label"
    )

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            size=14,
            color="#111827"
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    city = (
        df.groupby("city")
        .size()
        .reset_index(name="Customer Count")
    )

    fig2 = px.bar(
        city,
        x="city",
        y="Customer Count",
        title="Customers by City"
    )

    fig2.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            size=14,
            color="#111827"
        ),
        margin=dict(l=20, r=20, t=60, b=20)
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

# ==========================
# TOP10
# ==========================

st.divider()

st.subheader("🔥 Top 10 Highest Risk Customers")

risk=df.copy()

X=risk.drop(
    columns=["customer_id","churn"]
)

X=pd.get_dummies(
    X,
    columns=["gender","city"],
    drop_first=True
)

for col in model.feature_names_in_:
    if col not in X.columns:
        X[col]=0

X=X[model.feature_names_in_]

risk["Risk Score"]=(
    model.predict_proba(X)[:,1]*100
).round(1)

top10=risk.sort_values(
    "Risk Score",
    ascending=False
).head(10)

st.dataframe(
    top10[
        [
            "customer_id",
            "city",
            "portfolio_value",
            "Risk Score"
        ]
    ],
    use_container_width=True
)

st.divider()

# ==========================
# EXECUTIVE SUMMARY
# ==========================

durum="Low"

if oran>15:
    durum="High"

elif oran>5:
    durum="Medium"

st.info(f"""

## 🧠 AI Executive Summary

• Customers : **{toplam_musteri}**

• Churn : **{toplam_churn}**

• Churn Rate : **%{oran:.1f}**

• Risk Level : **{durum}**

AI recommends prioritizing inactive
customers and customers with
high portfolio value.

""")

# ==========================
# SEKMELER
# ==========================

tab1,tab2=st.tabs([
    "👤 Customer Analysis",
    "📂 Batch Analysis"
])



# ==========================
# TEK MÜŞTERİ ANALİZİ
# ==========================

with tab1:

    st.subheader("🔍 Customer Analysis")

    customer_id = st.text_input(
        "Customer ID",
        placeholder="Example: 1005"
    )

    if st.button("🚀 Analyze Customer", use_container_width=True):

        if customer_id == "":
            st.warning("Please enter a Customer ID.")

        else:

            try:

                customer_id = int(customer_id)

                customer = df[
                    df["customer_id"] == customer_id
                ]

                if customer.empty:

                    st.error("❌ Customer not found.")

                else:

                    st.success("✅ Customer Found")

                    left, right = st.columns([1, 1])

                    # -------------------------
                    # CUSTOMER INFORMATION
                    # -------------------------

                    with left:

                        st.subheader("📋 Customer Information")

                        st.dataframe(
                            customer,
                            use_container_width=True
                        )


                    # -------------------------
                    # MODEL PREPARATION
                    # -------------------------

                    customer_ml = customer.copy()

                    customer_ml = customer_ml.drop(
                        columns=["customer_id", "churn"]
                    )

                    customer_ml = pd.get_dummies(
                        customer_ml,
                        columns=["gender", "city"],
                        drop_first=True
                    )


                    for col in model.feature_names_in_:

                        if col not in customer_ml.columns:
                            customer_ml[col] = 0


                    customer_ml = customer_ml[
                        model.feature_names_in_
                    ]


                    prediction = model.predict(
                        customer_ml
                    )[0]


                    probability = model.predict_proba(
                        customer_ml
                    )[0][1]


                    # -------------------------
                    # AI PREDICTION
                    # -------------------------

                    with right:

                        st.subheader("🤖 AI Prediction")


                        gauge = go.Figure(

                            go.Indicator(

                                mode="gauge+number",

                                value=probability * 100,

                                number={
                                    "suffix": "%",
                                    "font": {
                                        "color": "white",
                                        "size": 40
                                    }
                                },

                                title={
                                    "text": "Customer Risk Score",
                                    "font": {
                                        "color": "white",
                                        "size": 20
                                    }
                                },

                                gauge={

                                    "axis": {
                                        "range": [0, 100],
                                        "tickcolor": "white"
                                    },

                                    "bar": {
                                        "color": "#007AFF"
                                    },

                                    "bgcolor": "#111111",

                                    "bordercolor": "#555555",

                                    "steps": [

                                        {
                                            "range": [0, 40],
                                            "color": "#166534"
                                        },

                                        {
                                            "range": [40, 70],
                                            "color": "#A16207"
                                        },

                                        {
                                            "range": [70, 100],
                                            "color": "#991B1B"
                                        }

                                    ]
                                }
                            )
                        )



                        gauge.update_layout(

                            height=320,

                            paper_bgcolor="#111111",

                            plot_bgcolor="#111111",

                            font={
                                "color": "white"
                            },

                            margin=dict(
                                l=20,
                                r=20,
                                t=50,
                                b=20
                            )
                        )


                        st.plotly_chart(

                            gauge,

                            use_container_width=True

                        )


                        if prediction == 1:

                            st.error("🔴 HIGH CHURN RISK")

                        else:

                            st.success("🟢 LOW CHURN RISK")                    


                    st.markdown("---")


                    st.subheader("🧠 AI Insight")


                    if prediction == 1:

                        st.warning(f"""
Customer behavior indicates a high churn probability.

Current Risk Score: **{probability*100:.1f}%**

### Recommended Actions

• Contact customer immediately

• Offer a personalized investment campaign

• Assign an investment advisor

• Monitor future trading activity
""")

                    else:

                        st.success(f"""
Customer shows a low churn probability.

Current Risk Score: **{probability*100:.1f}%**

### Recommended Actions

• Maintain customer engagement

• Offer premium investment opportunities

• Monitor portfolio activity
""")

                    st.divider()

                    col1, col2 = st.columns(2)

                    with col1:

                        st.subheader("🧠 AI Risk Analysis")

                        riskler = []

                        if customer.iloc[0]["last_login_days"] > 30:
                            riskler.append("📱 Uzun süredir giriş yapmıyor.")

                        if customer.iloc[0]["last_trade_days"] > 30:
                            riskler.append("📉 Uzun süredir işlem yapmamış.")

                        if customer.iloc[0]["monthly_trade_count"] < 3:
                            riskler.append("📊 İşlem hacmi düşük.")

                        if customer.iloc[0]["campaign_click"] == 0:
                            riskler.append("🎯 Kampanyalarla etkileşim yok.")

                        if customer.iloc[0]["complaints"] > 0:
                            riskler.append("⚠️ Şikayet kaydı mevcut.")

                        if customer.iloc[0]["cash_out_3m"] > 50000:
                            riskler.append("💸 Son 3 ayda yüksek para çıkışı.")


                        if len(riskler) == 0:

                            st.success(
                                "Belirgin risk bulunamadı."
                            )

                        else:

                            for r in riskler:
                                st.write(r)



                    with col2:

                        st.subheader("💡 AI Recommendations")

                        oneriler = []


                        if customer.iloc[0]["last_login_days"] > 30:
                            oneriler.append("📲 Push bildirimi gönder.")

                        if customer.iloc[0]["last_trade_days"] > 30:
                            oneriler.append("📞 Danışman müşteriyi arasın.")

                        if customer.iloc[0]["campaign_click"] == 0:
                            oneriler.append("🎁 Kişiselleştirilmiş kampanya.")

                        if customer.iloc[0]["monthly_trade_count"] < 3:
                            oneriler.append("📈 Eğitim içerikleri öner.")

                        if customer.iloc[0]["cash_out_3m"] > 50000:
                            oneriler.append("💼 Portföy danışmanlığı öner.")


                        if len(oneriler) == 0:

                            st.success(
                                "Ek aksiyon gerekmiyor."
                            )

                        else:

                            for o in oneriler:
                                st.write(o)


            except ValueError:

                st.error(
                    "Lütfen sadece sayı giriniz."
                )


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )



# ==========================
# TOPLU ANALİZ
# ==========================
with tab2:

    st.subheader("📂 Toplu Churn Analizi")

    uploaded_file = st.file_uploader(
        "Excel Dosyası Yükle",
        type=["xlsx"]
    )

    if uploaded_file is not None:

        batch_df = pd.read_excel(uploaded_file)

        st.success(f"✅ {len(batch_df)} müşteri yüklendi.")

        if st.button("🚀 Toplu Analizi Başlat"):

            sonuc_df = batch_df.copy()

            # Model için veri hazırla
            X = sonuc_df.drop(columns=["customer_id", "churn"])

            X = pd.get_dummies(
                X,
                columns=["gender", "city"],
                drop_first=True
            )

            # Eksik sütunları tamamla
            for col in model.feature_names_in_:
                if col not in X.columns:
                    X[col] = 0

            X = X[model.feature_names_in_]

            # Tahminler
            probabilities = model.predict_proba(X)[:, 1]

            sonuc_df["Churn Probability"] = (
                probabilities * 100
            ).round(1)

            # Risk Seviyesi
            risk = []

            for p in probabilities:

                if p >= 0.70:
                    risk.append("🔴 Yüksek")

                elif p >= 0.40:
                    risk.append("🟡 Orta")

                else:
                    risk.append("🟢 Düşük")

            sonuc_df["Risk Level"] = risk

            # En riskliler üstte
            sonuc_df = sonuc_df.sort_values(
                by="Churn Probability",
                ascending=False
            )

            st.subheader("📊 Analiz Sonuçları")

            st.dataframe(
                sonuc_df,
                use_container_width=True
            )

            # ==========================
            # TOPLU ANALİZ ÖZETİ
            # ==========================

            toplam = len(sonuc_df)
            yuksek = len(
                sonuc_df[
                    sonuc_df["Risk Level"] == "🔴 Yüksek"
                ]
            )

            orta = len(
                sonuc_df[
                    sonuc_df["Risk Level"] == "🟡 Orta"
                ]
            )

            dusuk = len(
                sonuc_df[
                    sonuc_df["Risk Level"] == "🟢 Düşük"
                ]
            )

            st.markdown("### 📊 Toplu Analiz Özeti")

            k1, k2, k3, k4 = st.columns(4)

            with k1:
                st.metric("👥 Toplam", toplam)

            with k2:
                st.metric("🔴 Yüksek Risk", yuksek)

            with k3:
                st.metric("🟡 Orta Risk", orta)

            with k4:
                st.metric("🟢 Düşük Risk", dusuk)

            fig = px.pie(
                sonuc_df,
                names="Risk Level",
                title="Risk Dağılımı"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # ==========================
            # EXCEL İNDİR
            # ==========================

            sonuc_df.to_excel(
                "riskli_musteriler.xlsx",
                index=False
            )

            with open(
                "riskli_musteriler.xlsx",
                "rb"
            ) as file:

                st.download_button(
                    label="📥 Sonuçları Excel Olarak İndir",
                    data=file,
                    file_name="AI_Churn_Result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )