import mlflow.pyfunc
import json
import os

mlflow.set_tracking_uri("http://localhost:5001")

@st.cache_resource
def load_model_artifacts():
    model_name = "Ethereum_Fraud_XGBoost"
    stage = "Production"

    try:
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.pyfunc.load_model(model_uri)
        features = ['account_age_mins', 'avg_min_sent', 'avg_min_recv', 'eth_balance',
                    'erc20_total_tnx', 'erc20_unique_rec_addr', 'sent_tnx_count',
                    'recv_tnx_count', 'unique_sent_addr', 'unique_recv_addr']

        st.success(f"Modelo carregado do MLflow: {model_uri}")
        return model, features

    except Exception as e:
        st.warning(f"Não foi possível conectar ao MLflow: {e}")
        st.info("Tentando carregar arquivos locais de backup...")
        try:
            model = joblib.load('xgboost_fraud_detector.pkl')
            features = joblib.load('model_features.pkl')
            return model, features
        except:
            return None, None