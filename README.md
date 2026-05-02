# Ethereum Fraud Monitor — MLOps Pipeline

Sistema de detecção de fraudes em transações Ethereum com pipeline MLOps completo: treinamento automatizado via Apache Airflow, rastreamento de experimentos com MLflow e interface de monitoramento em tempo real com Streamlit.

---

## Visão Geral

O projeto implementa um fluxo de MLOps de ponta a ponta para classificar transações Ethereum como legítimas ou fraudulentas usando um modelo XGBoost. Todo o ambiente roda em containers Docker, tornando a execução simples e reproduzível.

```
Airflow (agendador) ──→ Treina XGBoost ──→ MLflow (registry)
                                                  │
                                        Promote para "Production"
                                                  │
                                        Streamlit (monitoring app)
                                                  │
                                        Simula transações em tempo real
                                        e exibe predições no dashboard
```

---

## Arquitetura

| Componente | Tecnologia | Porta |
|---|---|---|
| Orquestrador de DAGs | Apache Airflow 2.7.1 | 8081 |
| Tracking & Registry de modelos | MLflow 2.7.1 | 5001 |
| Banco de dados do Airflow | PostgreSQL 13 | 5432 |
| Dashboard de monitoramento | Streamlit | 8501 |

---

## Estrutura do Projeto

```
ml_ops_eth/
├── app/
│   ├── app.py                  # Dashboard Streamlit de monitoramento
│   ├── requirements.txt        # Dependências da aplicação
│   └── Dockerfile              # Imagem do Streamlit
├── dags/
│   └── dag_fraud_ethereum.py   # DAG de treinamento diário
├── plugins/                    # Plugins do Airflow (extensível)
├── logs/                       # Logs do Airflow (gerados em runtime)
├── mlartifacts/                # Artefatos dos modelos treinados
├── Dockerfile_airflow          # Imagem customizada do Airflow
├── docker-compose.yml          # Orquestração completa dos serviços
└── load_model_artifacts.py     # Utilitário de carregamento de artefatos
```

---

## Como Executar

### Pré-requisitos

- [Docker](https://docs.docker.com/get-docker/) e [Docker Compose](https://docs.docker.com/compose/install/)
- Portas **5432**, **5001**, **8081** e **8501** disponíveis

### 1. Subir todos os serviços

```bash
git clone <url-do-repositório>
cd ml_ops_eth
docker compose up -d
```

Aguarde cerca de 30–60 segundos para que todos os serviços inicializem. Você pode acompanhar com:

```bash
docker compose logs -f
```

### 2. Acessar as interfaces

| Interface | URL | Credenciais |
|---|---|---|
| Airflow (agendador) | http://localhost:8081 | `admin` / `admin` |
| MLflow (tracking) | http://localhost:5001 | — |
| Streamlit (dashboard) | http://localhost:8501 | — |

### 3. Treinar o primeiro modelo

O Airflow já agenda a DAG `pipeline_fraude_ethereum` para executar diariamente. Para treinar imediatamente:

1. Acesse o Airflow em http://localhost:8081
2. Localize a DAG **pipeline_fraude_ethereum**
3. Clique em **Trigger DAG** (ícone de play)
4. Aguarde a task `treinar_modelo_v1` ficar verde (sucesso)

### 4. Promover o modelo para produção

1. Acesse o MLflow em http://localhost:5001
2. Vá em **Models → Modelo_Fraude_Ethereum**
3. Selecione a versão mais recente
4. No campo **Stage**, selecione **Production** e confirme

### 5. Usar o dashboard de monitoramento

1. Acesse o Streamlit em http://localhost:8501
2. O painel lateral exibirá **MLflow: Conectado** quando o modelo estiver em produção
3. Ajuste a **Taxa de Injeção de Fraudes** (slider) para controlar quantas transações fraudulentas serão simuladas
4. Ajuste a **Velocidade de Simulação** (segundos entre transações)
5. Clique em **INICIAR** para começar a simulação em tempo real

### Parar os serviços

```bash
docker compose down
```

Para remover também os volumes (banco de dados e artefatos):

```bash
docker compose down -v
```

---

## Como Funciona

### Pipeline de Treinamento (Airflow DAG)

A DAG `pipeline_fraude_ethereum` executa a task `treinar_modelo_v1` diariamente e realiza as seguintes etapas:

1. **Geração de dados sintéticos** — cria 500 amostras com distribuição 80% legítimas / 20% fraudulentas
2. **Feature engineering** — quatro features numéricas descrevem cada transação:

| Feature | Descrição |
|---|---|
| `account_age_mins` | Antiguidade da conta em minutos |
| `avg_min_sent` | Valor médio enviado por transação |
| `avg_min_recv` | Valor médio recebido por transação |
| `eth_balance` | Saldo atual em ETH |

3. **Treinamento** — classifica transações com XGBClassifier (XGBoost)
4. **Registro no MLflow** — loga métricas, parâmetros e salva o modelo no registry como `Modelo_Fraude_Ethereum`

### Rastreamento de Experimentos (MLflow)

O MLflow centraliza:

- Histórico de runs de treinamento (métricas, parâmetros, artefatos)
- Versionamento de modelos
- Controle de estágio: **None → Staging → Production → Archived**

### Dashboard de Monitoramento (Streamlit)

O `app.py` conecta-se ao MLflow e carrega o modelo na stage `Production`. Em seguida:

1. Gera transações Ethereum aleatórias com probabilidade de fraude configurável
2. Executa `predict_proba` para obter a probabilidade de fraude de cada transação
3. Exibe em tempo real:
   - **Métricas gerais**: total de transações, fraudes bloqueadas, percentual de risco, status do modelo
   - **Gráfico temporal**: curva de risco com zona vermelha acima de 50%
   - **Log de transações**: tabela das últimas 20 com carteira, saldo, idade da conta, risco e status (verde/vermelho)
   - **Alerta visual**: banner vermelho quando fraude é detectada

---

## Desenvolvimento Local (sem Docker)

Para executar apenas o dashboard fora do Docker (exige um MLflow acessível):

```bash
cd app
pip install -r requirements.txt
MLFLOW_TRACKING_URI=http://localhost:5001 streamlit run app.py
```

---

## Tecnologias

- **Apache Airflow 2.7.1** — orquestração e agendamento do pipeline
- **MLflow 2.7.1** — rastreamento de experimentos e model registry
- **XGBoost** — modelo de classificação binária
- **scikit-learn** — utilitários de ML (train/test split, métricas)
- **Streamlit** — interface web interativa em Python
- **Plotly** — gráficos no dashboard
- **PostgreSQL 13** — backend de metadados do Airflow
- **Docker & Docker Compose** — containerização e orquestração de serviços
