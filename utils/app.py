import streamlit as st
import pandas as pd
import joblib
from utils.features import preparar_features

st.set_page_config(page_title="Previsão de Ondas de Calor", layout="wide")

st.title("🔥 Previsão de Ondas de Calor – Modelo V1")
st.write("Seu mini laboratório climático pessoal.")

# 1. Carregar modelo
modelo = joblib.load("modelo/modelo_heatwave.pkl")

# 2. Carregar dados GOLD
gold = pd.read_parquet("data/gold/dados_modelo_gold_PR.parquet")

# 3. Inputs do usuário
lat = st.number_input("Latitude", -90.0, 90.0, -24.5)
lon = st.number_input("Longitude", -180.0, 180.0, -53.5)
data = st.date_input("Data da previsão")

if st.button("Prever Onda de Calor"):
    entrada = preparar_features(lat, lon, data, gold)

    pred = modelo.predict(entrada)[0]
    proba = modelo.predict_proba(entrada)[0][1]

    st.subheader("Resultado")
    st.metric(
        "Onda de calor prevista?",
        "SIM" if pred == 1 else "NÃO",
    )
    st.metric(
        "Probabilidade estimada",
        f"{proba * 100:.2f}%"
    )

    st.write("Dados usados na previsão:")
    st.dataframe(entrada)
