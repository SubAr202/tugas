import streamlit as st
import joblib
import pandas as pd

# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Prediksi Berat Badan",
    page_icon="⚖️",
    layout="centered"
)

# =============================
# CUSTOM CSS (MODERN UI)
# =============================
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.main {
    background-color: #0f172a;
}
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #38bdf8;
}

.subtitle {
    text-align: center;
    color: #cbd5f5;
    margin-bottom: 25px;
}

.result-box {
    background: linear-gradient(135deg, #1e293b, #020617);
    padding: 18px;
    border-radius: 14px;
    margin-bottom: 12px;
    font-size: 18px;
    font-weight: 600;
    color: #e5e7eb;
}

.stButton > button {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: black;
    font-weight: bold;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
}

label {
    color: #e5e7eb !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
model_logreg = joblib.load("model_logreg.pkl")
model_svm = joblib.load("model_svm.pkl")
model_nb = joblib.load("model_nb.pkl")
le = joblib.load("label_encoder.pkl")

# =============================
# HEADER
# =============================
st.markdown('<div class="title">⚖️ Prediksi Berat Badan</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning Classification</div>', unsafe_allow_html=True)

# =============================
# INPUT CARD
# =============================
st.markdown('<div class="card">', unsafe_allow_html=True)

gender = st.selectbox("👤 Gender", ["L", "P"])
tinggi = st.number_input("📏 Tinggi Badan (cm)", 140, 200)
berat = st.number_input("⚖️ Berat Badan (kg)", 35, 150)

imt = berat / ((tinggi / 100) ** 2)

st.markdown("</div>", unsafe_allow_html=True)

# =============================
# PREDICTION
# =============================

def berat_badan_ideal(tinggi_cm):
    tinggi_m = tinggi_cm / 100
    bb_min = 18.5 * (tinggi_m ** 2)
    bb_max = 24.9 * (tinggi_m ** 2)
    return round(bb_min, 1), round(bb_max, 1)

if st.button("🔍 Prediksi"):
    g = 0 if gender == "L" else 1
    data = pd.DataFrame([[g, tinggi, berat, imt]],
                         columns=['gender', 'tinggi', 'berat', 'imt'])

    pred_logreg = le.inverse_transform(model_logreg.predict(data))[0]
    pred_svm = le.inverse_transform(model_svm.predict(data))[0]
    pred_nb = le.inverse_transform(model_nb.predict(data))[0]

    # berat badan ideal
    bb_min, bb_max = berat_badan_ideal(tinggi)

    # status & rekomendasi berdasarkan IMT
    if imt < 18.5:
        status = "Kurus"
        rekomendasi = f"🔺 Tambah berat ± {round(bb_min - berat, 1)} kg"
    elif imt <= 24.9:
        status = "Normal"
        rekomendasi = "✅ Berat badan sudah ideal"
    elif imt <= 29.9:
        status = "Overweight"
        rekomendasi = f"🔻 Kurangi berat ± {round(berat - bb_max, 1)} kg"
    else:
        status = "Obesitas"
        rekomendasi = f"🔻 Kurangi berat ± {round(berat - bb_max, 1)} kg"

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Hasil Prediksi")

    st.markdown(f"""
        <div class="result-box">📌 Logistic Regression : <b>{pred_logreg}</b></div>
        <div class="result-box">📏 IMT : <b>{imt:.2f}</b> ({status})</div>
        <div class="result-box">⚖️ Berat Ideal : <b>{bb_min} – {bb_max} kg</b></div>
        <div class="result-box">{rekomendasi}</div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
