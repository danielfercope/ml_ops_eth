import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import plotly.express as px
from datetime import datetime
import mlflow.pyfunc
import os

st.set_page_config(
    page_title="Ethereum Fraud Monitor (MLOps)",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .fraud-alert {
        background-color: #ff4b4b;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .safe-status {
        background-color: #00cc66;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

#MLOPS
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)


@st.cache_resource
def load_model_from_mlflow():
    model_name = "Modelo_Fraude_Ethereum"
    stage = "Production"

    try:
        model_uri = f"models:/{model_name}/{stage}"
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        features_esperadas = ['account_age_mins', 'avg_min_sent', 'avg_min_recv', 'eth_balance']

        return loaded_model, features_esperadas, None

    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"

model, model_features, model_load_error = load_model_from_mlflow()

def generate_random_transaction(fraud_probability):
    is_suspicious = random.random() < fraud_probability

    data = {
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'address': f"0x{random.getrandbits(32):08x}...",
        # Features Numéricas Principais
        'account_age_mins': random.uniform(0, 100) if is_suspicious else random.uniform(1000, 500000),
        'avg_min_sent': random.uniform(0, 10) if is_suspicious else random.uniform(50, 10000),
        'avg_min_recv': random.uniform(0, 10) if is_suspicious else random.uniform(50, 10000),
        'eth_balance': random.uniform(0, 0.5) if is_suspicious else random.uniform(0, 100),
        'contracts_created_count': 0,
        'erc20_total_tnx': random.randint(0, 5) if is_suspicious else random.randint(0, 500),
    }
    return data, is_suspicious


def prepare_for_prediction(data_dict, feature_columns):
    df = pd.DataFrame([data_dict])
    df_final = df[feature_columns]
    return df_final

if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame()
if 'simulation_running' not in st.session_state:
    st.session_state['simulation_running'] = False
if 'fraud_count' not in st.session_state:
    st.session_state['fraud_count'] = 0
if 'total_count' not in st.session_state:
    st.session_state['total_count'] = 0

st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/05/Ethereum_logo_2014.svg/1200px-Ethereum_logo_2014.svg.png",
    width=50)
st.sidebar.title("Controles MLOps")

st.sidebar.markdown("**Status do MLflow:**")
if model:
    st.sidebar.success(f"Conectado! Modelo v.Prod carregado.")
else:
    st.sidebar.error("Desconectado ou Modelo não encontrado em Produção.")
    if model_load_error:
        st.sidebar.caption(f"URI: {mlflow_uri}")
        with st.sidebar.expander("Detalhes do erro"):
            st.code(model_load_error, language="text")
    if st.sidebar.button("Tentar reconectar"):
        st.cache_resource.clear()
        st.rerun()

st.sidebar.markdown("---")

fraud_rate = st.sidebar.slider("Taxa de Injeção de Fraudes (%)", 0, 100, 15) / 100
speed = st.sidebar.slider("Velocidade (segundos)", 0.1, 5.0, 1.0)

col_btn1, col_btn2 = st.sidebar.columns(2)
if col_btn1.button("INICIAR"):
    st.session_state['simulation_running'] = True
if col_btn2.button("PARAR"):
    st.session_state['simulation_running'] = False

if st.sidebar.button("Limpar Histórico"):
    st.session_state['history'] = pd.DataFrame()
    st.session_state['fraud_count'] = 0
    st.session_state['total_count'] = 0

# --- LAYOUT PRINCIPAL ---
st.title("Monitor de Fraudes Ethereum")

# Métricas
m1, m2, m3, m4 = st.columns(4)
m1.metric("Transações Analisadas", st.session_state['total_count'])
m2.metric("Fraudes Bloqueadas", st.session_state['fraud_count'], delta_color="inverse")

fraud_percentage = (st.session_state['fraud_count'] / st.session_state['total_count'] * 100) if st.session_state[
                                                                                                    'total_count'] > 0 else 0
m3.metric("% de Risco", f"{fraud_percentage:.1f}%")

status_html = "<b>ONLINE 🟢</b>" if model else "<b>OFFLINE 🔴</b>"
m4.markdown(f"<div style='text-align:center; padding: 10px;'>Status do Modelo:<br>{status_html}</div>",
            unsafe_allow_html=True)

alert_placeholder = st.empty()
col_chart, col_table = st.columns([1, 2])

with col_chart:
    st.subheader("Probabilidade de Fraude")
    chart_placeholder = st.empty()

with col_table:
    st.subheader("Log de Transações")
    table_placeholder = st.empty()

if st.session_state['simulation_running']:

    if model is None:
        st.error(
            "Não é possível iniciar: Modelo não carregado do MLflow. Verifique se a DAG rodou e se o modelo foi promovido para 'Production'.")
        st.session_state['simulation_running'] = False
    else:
        raw_data, _ = generate_random_transaction(fraud_probability=fraud_rate)

        try:
            input_df = prepare_for_prediction(raw_data, model_features)
            prediction = model.predict(input_df)[0]

            try:
                proba_all = model.predict_proba(input_df)
                proba = proba_all[0][1]
            except:
                proba = float(prediction)

            st.session_state['total_count'] += 1
            if prediction == 1:
                st.session_state['fraud_count'] += 1
                alert_placeholder.markdown(
                    f"<div class='fraud-alert'>🚨 FRAUDE DETECTADA! Carteira {raw_data['address']} bloqueada! (Score: {proba:.2f})</div>",
                    unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            # 4. Atualizar Histórico
            new_row = {
                'Hora': raw_data['timestamp'],
                'Carteira': raw_data['address'],
                'Saldo (ETH)': f"{raw_data['eth_balance']:.4f}",
                'Idade (min)': f"{raw_data['account_age_mins']:.1f}",
                'Risco (%)': f"{proba * 100:.1f}%",
                'Status': '🔴 FRAUDE' if prediction == 1 else '🟢 LEGÍTIMA'
            }

            current_df = st.session_state['history']
            new_df = pd.DataFrame([new_row])
            st.session_state['history'] = pd.concat([new_df, current_df], ignore_index=True).head(20)

            def color_fraud(val):
                color = '#ffcccc' if val == '🔴 FRAUDE' else '#ccffcc'
                return f'background-color: {color}'


            styled_df = st.session_state['history'].style.map(color_fraud, subset=['Status'])
            table_placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)

            if not st.session_state['history'].empty:
                df_chart = st.session_state['history'].copy()
                df_chart['Risco_Float'] = df_chart['Risco (%)'].str.replace('%', '').astype(float) / 100
                fig = px.line(df_chart, x='Hora', y='Risco_Float', title="Oscilação de Risco", markers=True)
                fig.update_yaxes(range=[0, 1])
                fig.add_hrect(y0=0.5, y1=1.0, line_width=0, fillcolor="red", opacity=0.1)
                chart_placeholder.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Erro na inferência: {e}")
            st.session_state['simulation_running'] = False

        time.sleep(speed)
        st.rerun()

else:
    if not st.session_state['history'].empty:
        def color_fraud(val):
            color = '#ffcccc' if val == '🔴 FRAUDE' else '#ccffcc'
            return f'background-color: {color}'


        styled_df = st.session_state['history'].style.map(color_fraud, subset=['Status'])
        table_placeholder.dataframe(styled_df, use_container_width=True, hide_index=True)