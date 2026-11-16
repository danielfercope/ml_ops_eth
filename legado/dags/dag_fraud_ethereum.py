from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient
import random

mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("deteccao_fraude_eth")

def treinar_modelo():
    data = []
    for _ in range(500):
        is_fraud = random.random() < 0.2
        row = {
            'account_age_mins': random.uniform(0, 100) if is_fraud else random.uniform(1000, 500000),
            'avg_min_sent': random.uniform(0, 10) if is_fraud else random.uniform(50, 10000),
            'avg_min_recv': random.uniform(0, 10) if is_fraud else random.uniform(50, 10000),
            'eth_balance': random.uniform(0, 0.5) if is_fraud else random.uniform(0, 100),
            'target': 1 if is_fraud else 0
        }
        data.append(row)

    df = pd.DataFrame(data)
    X = df.drop('target', axis=1)
    y = df['target']

    with mlflow.start_run():
        model = xgb.XGBClassifier()
        model.fit(X, y)

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="Modelo_Fraude_Ethereum"
        )

        # Promove automaticamente a versão recém-registrada para Production
        client = MlflowClient()
        latest = client.get_latest_versions("Modelo_Fraude_Ethereum", stages=["None"])
        if latest:
            client.transition_model_version_stage(
                name="Modelo_Fraude_Ethereum",
                version=latest[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            print(f"Modelo v{latest[0].version} promovido para Production!")
        else:
            print("Modelo registrado, mas versão não encontrada para promoção.")


with DAG('pipeline_fraude_ethereum', start_date=datetime(2023, 1, 1), schedule_interval='@daily', catchup=False) as dag:
    task_treino = PythonOperator(
        task_id='treinar_modelo_v1',
        python_callable=treinar_modelo
    )