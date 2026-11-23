# MLOps Ethereum — detecção de fraude com ciclos agendados

Pipeline MLOps do Trabalho N3. Um **único workflow agendado** no GitHub Actions incorpora lotes do
[Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset)
(Kaggle, ODbL; veja [`data/README.md`](data/README.md)). A cada ciclo ele elege as features por votação de
3 métodos, treina 2 modelos com as mesmas features, registra tudo no MLflow e **só troca o modelo vigente
se o novo superar o limiar** fixado em [`config/config.yaml`](config/config.yaml).

```text
           schedule (cron) / workflow_dispatch
                         │
             .github/workflows/pipeline_mlops.yml
                         │  CYCLE_ID = AAAAMMDD-HHMM-r<run>
                         ▼
                   run_pipeline.sh  ─────────────►  evidencias/<cycle_id>/  (notebooks executados, log, resumo)
                         │
  01_etl ──► 02_votacao_features ──► 03_treino_competicao ──► 04_selecao_promocao ──► 05_previsoes
  lotes,      3 métodos de seleção    A = Regressão Logística   vencedor vs vigente      vigente prevê
  partições,  + votação → features    B = XGBoost (mesmas       regra do config →        o teste e
  data_version  eleitas                features e partições)     promovido | mantido      compara ciclos
                         │
                         ▼   tudo verde?  ──não──►  nada é commitado (sem promoção pela metade)
                        sim
                         ▼
   commit do estado: mlflow.db, mlartifacts/, models/, artifacts/, data/processed/lotes.json, evidencias/
                         ▼
               painel (Dev 5) → GitHub Pages
```

| Notebook | Frente | O que faz | Saídas principais |
|---|---|---|---|
| [`01_etl`](notebooks/01_etl.ipynb) | Dev 1 | ingere `data/raw/lote_*.csv`, remove duplicados, partição fixa por hash do `Address` (60/20/20), `data_version` | `data/processed/*.parquet`, `schema.json`, `lotes.json` |
| [`02_votacao_features`](notebooks/02_votacao_features.ipynb) | Dev 2 | SelectKBest, Random Forest e RFE; eleita com ≥ 2 de 3 votos | `artifacts/<cycle_id>/features_eleitas.json`, tabela de votos |
| `03_treino_competicao` | Dev 3 | treina A e B no mesmo treino, avalia na mesma validação | `artifacts/<cycle_id>/competicao.json`, um run por modelo |
| `04_selecao_promocao` | Dev 3 | `F1_val ≥ F1_vigente + 0,01` e `F1_val ≥ 0,60` → promove | `models/modelo_vigente.json`, `models/historico_decisoes.csv` |
| [`05_previsoes`](notebooks/05_previsoes.ipynb) | Dev 4 | carrega o vigente pelo `run_id`, prevê o teste e mostra o que mudou entre os ciclos | `artifacts/<cycle_id>/previsoes.csv`, run `previsoes` |

Todo run do MLflow leva as tags `cycle_id`, `data_version` e `etapa` (`etl`, `votacao_features`,
`treino_competicao`, `promocao`, `previsoes`). Veja [`src/rastreio.py`](src/rastreio.py).

---

## Instalação

Requer **Python 3.12** (a mesma versão do workflow). As versões em `requirements.txt` são exatas: o modelo
é serializado com cloudpickle e só carrega de volta com as mesmas versões de scikit-learn e XGBoost.

```bash
git clone https://github.com/danielfercope/ml_ops_eth.git
cd ml_ops_eth
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q          # testes unitários
```

Os lotes já estão versionados em `data/raw/` e `data/reserva/`, então não é preciso baixar nada do Kaggle.
Para regerar os lotes a partir do CSV original: `python -m src.ingestao gerar-lotes`.

## Execução local

```bash
./run_pipeline.sh                              # um ciclo completo, CYCLE_ID = AAAAMMDD-HHMM-local
CYCLE_ID=20261110-0900-r12 ./run_pipeline.sh   # com um cycle_id escolhido
```

O script é o mesmo que o workflow executa: roda 01 → 05 com `jupyter nbconvert --execute`, para no
primeiro erro e grava os notebooks executados e o `pipeline.log` em `evidencias/<cycle_id>/`.
Os notebooks também podem ser abertos e executados um a um no Jupyter, a partir de `notebooks/`.

> **Não commite o estado de uma execução local.** O ciclo local altera `mlflow.db`, `mlartifacts/`,
> `models/` e `data/processed/lotes.json`, que são o histórico oficial dos ciclos agendados.
> Para descartar: `git restore mlflow.db models data/processed/lotes.json && git clean -fd mlartifacts`.
> Se quiser um MLflow separado, use `MLFLOW_TRACKING_URI=sqlite:////tmp/mlflow_local.db ./run_pipeline.sh`.
> As pastas `artifacts/*-local/` e `evidencias/*-local/` já ficam fora do git.

## Execução agendada (GitHub Actions)

[`.github/workflows/pipeline_mlops.yml`](.github/workflows/pipeline_mlops.yml):

- **Gatilhos:** `schedule` (`0 12 * * *`, todo dia às 09:00 de Brasília) e `workflow_dispatch` (botão
  *Run workflow* na aba Actions). O `schedule` só dispara na **branch padrão** e pode atrasar alguns minutos.
- **Um job sequencial** com `concurrency: pipeline-mlops`, então dois ciclos nunca rodam juntos.
- **Passos:** checkout → Python 3.12 com cache do pip → `pip install -r requirements.txt` → `pytest` →
  `run_pipeline.sh` → exportar os JSONs do painel (`src/exportar_painel.py`, se existir) → resumo do ciclo
  no *job summary* → commit do estado → upload de `evidencias/`, `models/`, `artifacts/` e `mlflow.db` como
  artifact (também em caso de falha, para depurar).
- **Persistência entre ciclos:** o próprio workflow commita `mlflow.db`, `mlartifacts/`, `models/`,
  `artifacts/`, `data/processed/lotes.json` e `evidencias/<cycle_id>/` na branch padrão
  (commit `🤖 Ciclo <cycle_id>: <decisão>`). Se qualquer notebook falha, o job para antes do commit e o
  repositório continua no ciclo anterior.
- **Painel:** um segundo job publica o build do `web/` no GitHub Pages. Ele usa `continue-on-error`, então
  nunca deixa o ciclo vermelho.

Configuração única do repositório (*Settings*):

1. *Actions → General → Workflow permissions*: **Read and write permissions**.
2. *Pages → Source*: **GitHub Actions** (só para o painel).
3. Se a branch padrão tiver proteção, permitir o push do `github-actions[bot]`.

### Painel local

```bash
python -m src.exportar_painel   # gera public/data/ a partir do estado do pipeline (valida o contrato)
npm ci
npm run dev                     # http://localhost:5173/ (dados reais; ?cenario=padrao|mantido|vazio para os mocks)
npm test                        # Vitest: as 5 telas renderizadas com os mocks
```

Contrato dos JSONs e cenários mock: [`docs/painel_dados.md`](docs/painel_dados.md).

### Demonstração dos ciclos

| Ciclo | O que fazer antes | Resultado esperado |
|---|---|---|
| 1 | nada: `data/raw/` só tem o `lote_01` | `promovido` (ainda não há vigente) |
| 2 | liberar o lote 2 e commitar (abaixo) | nova votação com `data_version v2-…`, `promovido` ou `mantido` com o motivo |
| 3 (opcional) | nada (sem dados novos) | mesmo `data_version`, treino idêntico → `mantido` |

```bash
git pull                                           # traz o commit do ciclo 1 feito pelo workflow
git mv data/reserva/lote_02.csv data/raw/lote_02.csv
git commit -m "Libera o lote_02 para o ciclo 2" && git push
```

## MLflow

```bash
python -m src.rastreio ui                          # ajusta caminhos e abre http://127.0.0.1:5000
# equivalente a:
python -m src.rastreio preparar
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

O tracking é um SQLite (`mlflow.db`) e os artefatos ficam em `mlartifacts/`, ambos na raiz e commitados.
O MLflow grava **caminhos absolutos** dos artefatos no banco (no runner, `/home/runner/work/...`).
`python -m src.rastreio preparar`, que o `run_pipeline.sh` também chama, reescreve esses caminhos para
a pasta atual, para o mesmo `mlflow.db` funcionar em qualquer máquina. Filtre os runs de um ciclo com
`tags.cycle_id = '<cycle_id>'`.

## Reproduzir as previsões

```python
import json, pandas as pd, mlflow
from src.rastreio import configurar_mlflow

configurar_mlflow()                                    # sqlite:///<raiz>/mlflow.db
man = json.load(open("models/modelo_vigente.json"))   # manifesto do vigente (Dev 3)
modelo = mlflow.sklearn.load_model(man["model_uri"])  # Pipeline: pré-processamento → features eleitas → estimador
teste = pd.read_parquet("data/processed/teste.parquet")
proba = modelo.predict_proba(teste)[:, 1]
```

Ou rode só o notebook `05_previsoes.ipynb`. É preciso que `data/processed/` exista (`python -m src.preprocessamento`).
O F1 de teste calculado ali deve bater com `f1_teste` do manifesto.

---

## Estrutura

```text
ml_ops_eth/
├── .github/workflows/pipeline_mlops.yml   # workflow agendado (Dev 4)
├── run_pipeline.sh                        # 01 → 05, usado pelo workflow e localmente (Dev 4)
├── config/config.yaml                     # métrica, limiar, regra de votação, seeds (fixados antes do ciclo 1)
├── data/
│   ├── raw/                               # lotes visíveis ao pipeline
│   ├── reserva/                           # lote_02, liberado antes do ciclo 2
│   └── processed/                         # partições (fora do git) + lotes.json (commitado)
├── notebooks/                             # 01 a 05
├── src/
│   ├── ingestao.py, preprocessamento.py   # Dev 1
│   ├── features.py                        # Dev 2
│   ├── modelagem.py, promocao.py          # Dev 3
│   ├── rastreio.py, previsoes.py          # Dev 4
│   └── exportar_painel.py                 # Dev 5
├── artifacts/<cycle_id>/                  # features eleitas, competição, decisão e previsões do ciclo
├── models/                                # modelo_vigente.json e historico_decisoes.csv
├── mlflow.db, mlartifacts/                # tracking do MLflow (persistido pelo workflow)
├── evidencias/<cycle_id>/                 # notebooks executados, pipeline.log, resumo_ciclo.json
├── web/, index.html                       # painel: app Vite + React + TS (Dev 5)
├── prototipo.html                         # protótipo aprovado do painel (HTML único)
├── public/data/                           # JSONs do painel gerados por exportar_painel.py
├── docs/                                  # seções do relatório por frente
├── reports/                               # relatorio_n3.pdf (python reports/gerar_relatorio.py)
├── tests/
└── legado/                                # Airflow + Docker + Streamlit (versão anterior, fora do fluxo)
```

## Tecnologias

Python 3.12 · pandas · scikit-learn · XGBoost · MLflow (SQLite) · Jupyter/nbconvert · GitHub Actions ·
Vite + React + TypeScript (painel). A versão anterior com Airflow, Docker e Streamlit está em
[`legado/`](legado/README.md).
