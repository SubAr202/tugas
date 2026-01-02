import streamlit as st
import joblib
import pandas as pd

# Load model
model_logreg = joblib.load("model_logreg.pkl")
model_svm = joblib.load("model_svm.pkl")
model_nb = joblib.load("model_nb.pkl")
le = joblib.load("label_encoder.pkl")

st.title("Prediksi BMI dengan Machine Learning")

gender = st.selectbox("Gender", ["L", "P"])
tinggi = st.number_input("Tinggi Badan (cm)", 140, 200)
berat = st.number_input("Berat Badan (kg)", 35, 150)

imt = berat / ((tinggi / 100) ** 2)

if st.button("Prediksi"):
    g = 0 if gender == "L" else 1
    data = pd.DataFrame([[g, tinggi, berat, imt]],
                         columns=['gender', 'tinggi', 'berat', 'imt'])

    pred_logreg = le.inverse_transform(model_logreg.predict(data))[0]
    pred_svm = le.inverse_transform(model_svm.predict(data))[0]
    pred_nb = le.inverse_transform(model_nb.predict(data))[0]

    st.subheader("Hasil Prediksi:")
    st.write("📌 Logistic Regression:", pred_logreg)
    st.write("📌 SVM:", pred_svm)
    st.write("📌 Naive Bayes:", pred_nb)
