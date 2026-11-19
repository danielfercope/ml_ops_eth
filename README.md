# Ethereum Fraud Monitor — MLOps Pipeline

Sistema de detecção de fraudes em transações Ethereum com pipeline MLOps completo: treinamento automatizado via Apache Airflow, rastreamento de experimentos com MLflow e interface de monitoramento em tempo real com Streamlit.

---

## Visão Geral

O projeto implementa um fluxo de MLOps de ponta a ponta para classificar transações Ethereum como legítimas ou fraudulentas usando um modelo XGBoost. Todo o ambiente roda em containers Docker, tornando a execução simples e reproduzível.

```md
Airflow (agendador) ──> Treina XGBoost ──> MLflow (registry)
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
| --- | --- | --- |
| Orquestrador de DAGs | Apache Airflow 2.7.1 | 8081 |
| Tracking & Registry de modelos | MLflow 2.7.1 | 5001 |
| Banco de dados do Airflow | PostgreSQL 13 | 5432 |
| Dashboard de monitoramento | Streamlit | 8501 |

---

## Estrutura do Projeto

```md
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
├── web/                        # Painel de acompanhamento (Vite + React + TS)
├── public/                     # Arquivos estáticos do painel
├── index.html                  # Painel de acompanhamento (página principal, lê os mocks)
├── package.json                # Scripts e dependências do painel
├── vite.config.ts              # Configuração do Vite
├── Dockerfile_airflow          # Imagem customizada do Airflow
└── docker-compose.yml          # Orquestração completa dos serviços
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
| --- | --- | --- |
| Airflow (agendador) | <http://localhost:8081> | `admin` / `admin` |
| MLflow (tracking) | <http://localhost:5001> | — |
| Streamlit (dashboard) | <http://localhost:8501> | — |

### 3. Treinar o primeiro modelo (automático)

Logo após o `docker compose up -d`, o próprio container do Airflow:

1. Despausa a DAG `pipeline_fraude_ethereum`
2. Dispara uma execução manual
3. A task `treinar_modelo_v1` registra o modelo no MLflow como `Modelo_Fraude_Ethereum`
4. A própria DAG promove a versão recém-criada para o stage **Production** (chamando `MlflowClient.transition_model_version_stage`)

Acompanhe pelo Airflow em <http://localhost:8081> até a task ficar verde (~30s após o stack subir). Se quiser executar novamente, use **Trigger DAG** na interface.

> **Observação:** se preferir gerenciar o estágio manualmente, basta editar a DAG `dags/dag_fraud_ethereum.py` e remover o bloco `client.transition_model_version_stage(...)`. A promoção poderá então ser feita pelo MLflow em **Models → Modelo_Fraude_Ethereum → versão mais recente → Stage → Production**.

### 4. Usar o dashboard de monitoramento

1. Acesse o Streamlit em <http://localhost:8501>
2. O painel lateral exibirá **MLflow: Conectado** quando o modelo estiver em produção
3. Ajuste a **Taxa de Injeção de Fraudes** (slider) para controlar quantas transações fraudulentas serão simuladas
4. Ajuste a **Velocidade de Simulação** (segundos entre transações)
5. Clique em **INICIAR** para começar a simulação em tempo real

### 5. Parar os serviços

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
| --- | --- |
| `account_age_mins` | Antiguidade da conta em minutos |
| `avg_min_sent` | Valor médio enviado por transação |
| `avg_min_recv` | Valor médio recebido por transação |
| `eth_balance` | Saldo atual em ETH |

3. **Treinamento** — classifica transações com XGBClassifier (XGBoost)
4. **Registro no MLflow** — loga métricas, parâmetros e salva o modelo no registry como `Modelo_Fraude_Ethereum`
5. **Promoção automática** — a versão recém-registrada é movida para o stage **Production** e versões anteriores são automaticamente arquivadas

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

## Painel de Acompanhamento (`web/`) — em desenvolvimento

Recurso adicional do trabalho (critério 10), frente do Dev 5 em [`docs/divisao_trabalho.pdf`](docs/divisao_trabalho.pdf). É um painel estático que mostra a evolução dos ciclos (modelo vigente, votação de features, competição e previsões) sem abrir o MLflow nem os notebooks. Ele será gerado pelo próprio pipeline e publicado no GitHub Pages, sem API nem servidor.

Hoje a página principal (`index.html`) é o protótipo navegável das 5 telas: HTML único, sem dependências, lendo **só os dados mock** de `web/mocks/<cenario>/` (formato em [`docs/painel_dados.md`](docs/painel_dados.md)). No cabeçalho dá para trocar o cenário (`padrao`, `mantido`, `vazio`) e forçar os estados de carregamento e erro. O app em React (`web/`) vai substituí-lo depois que o grupo aprovar o protótipo e os design tokens.

### Executar localmente

Requer Node.js 20.19+ ou 22.12+. Na raiz do repositório:

```bash
npm install
npm run dev      # painel em http://localhost:5173 (ex.: /?cenario=mantido#/votacao)
npm run build    # checagem de tipos (tsc) + build de produção em dist/ (inclui os mocks)
npm run preview  # serve o dist/ para conferir o build
npm run lint     # ESLint
```

### Progresso

- [x] Projeto Vite + React + TypeScript criado em `web/` (ainda com a tela inicial do template, fora da página principal)
- [x] Formato dos JSONs do painel (`ciclos.json`, `votacao.json`, `competicao.json`, `vigente.json`, `previsoes.json`) e versões mock
- [ ] Protótipo navegável com dados fictícios e design tokens aprovados pelo grupo (versão inicial em `index.html`, aguardando aprovação)
- [ ] React Router, layout com navegação e componentes compartilhados (cartão de KPI, tabela, selo `promovido`/`mantido`, estados de carregamento e erro)
- [ ] Telas: Visão geral, Ciclos, Votação, Competição e Previsões
- [ ] `src/exportar_painel.py` gerando os JSONs a partir do `mlflow.db`, do manifesto e do histórico de decisões
- [ ] Build e deploy no GitHub Pages dentro do workflow, junto com o Dev 4
- [ ] Testes de renderização com Vitest usando os JSONs mock

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
- **Typescript, React, Vite & ESLint** — aplicação e desenvolvimento web
