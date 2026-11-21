# Arquitetura do fluxo único e evidências de execução (Dev 4)

> Base para as seções "Arquitetura do fluxo único" e "Evidências de execução" do relatório N3
> (critérios 6, 7, 8 e 9).
> Código: `.github/workflows/pipeline_mlops.yml`, `run_pipeline.sh`, `src/rastreio.py`, `src/previsoes.py`.
> Notebook: `notebooks/05_previsoes.ipynb`. Testes: `tests/test_rastreio.py`, `tests/test_previsoes.py`.

## Arquitetura do fluxo único

### Uma execução = um ciclo
O projeto tem **um único fluxo**: o workflow `pipeline_mlops.yml` do GitHub Actions. Cada execução é um
**ciclo**, identificado por um `cycle_id` no formato `AAAAMMDD-HHMM-r<run_number>` (ex.: `20261110-0900-r12`).
O workflow gera esse id uma vez e o exporta como variável de ambiente `CYCLE_ID`. Os cinco notebooks o
leem por `src.rastreio.obter_cycle_id()`, então todos os runs, arquivos e commits do ciclo levam o mesmo id.

```text
gatilho: schedule (cron 0 12 * * *)  ou  workflow_dispatch
   │
   ├─ checkout · Python 3.12 (cache do pip) · pip install -r requirements.txt · pytest
   ├─ CYCLE_ID = AAAAMMDD-HHMM-r<run_number>
   ├─ run_pipeline.sh
   │     ├─ python -m src.rastreio preparar        (experimento MLflow + caminhos do mlflow.db)
   │     ├─ 01_etl ─► 02_votacao_features ─► 03_treino_competicao ─► 04_selecao_promocao ─► 05_previsoes
   │     │      (jupyter nbconvert --execute; para no primeiro erro)
   │     └─ evidencias/<cycle_id>/ : notebooks executados, pipeline.log, resumo_ciclo.json
   ├─ src/exportar_painel.py (Dev 5; nunca bloqueia o ciclo)
   ├─ resumo do ciclo no job summary
   ├─ commit do estado na branch padrão  ──►  "🤖 Ciclo <cycle_id>: <decisão>"
   ├─ upload de evidências como artifact (sempre, inclusive em falha)
   └─ job "painel": build do web/ e deploy no GitHub Pages (continue-on-error)
```

| Decisão | Escolha | Por quê |
|---|---|---|
| Orquestrador | GitHub Actions (`schedule` + `workflow_dispatch`) | dispensa servidor ligado; o histórico de execuções com o gatilho `schedule` é a evidência do critério 7. Airflow, Docker e Streamlit foram para `legado/` |
| Unidade de execução | notebooks executados com `nbconvert --execute` | o mesmo artefato serve de código, de relatório e de evidência (o notebook executado guarda as saídas do ciclo) |
| Script único | `run_pipeline.sh` | workflow e execução local rodam exatamente a mesma sequência; não há "versão de CI" diferente |
| Concorrência | `concurrency: pipeline-mlops`, sem cancelamento | dois ciclos simultâneos poderiam promover sobre o mesmo manifesto |
| Tracking | MLflow com SQLite (`mlflow.db`) + `mlartifacts/` | sem servidor; o banco é um arquivo e pode ser versionado |
| Persistência | o workflow commita o estado de volta no repositório | o runner é descartado ao fim de cada execução; sem isso, o ciclo 2 não veria o MLflow nem o vigente do ciclo 1 |
| Reprodutibilidade | versões exatas em `requirements.txt`, Python 3.12, seeds no `config.yaml`, `PYTHONHASHSEED=0` | o modelo é serializado com cloudpickle e só carrega com as mesmas versões; duas simulações independentes dos ciclos deram as mesmas métricas |

### Persistência entre ciclos (critério 6)
Ao fim de um ciclo verde, o workflow commita na branch padrão:

| Caminho | Conteúdo | Dono |
|---|---|---|
| `mlflow.db` | todos os runs (6 por ciclo: `etl`, `votacao_features`, 2 × `treino_competicao`, `promocao`, `previsoes`) e o Model Registry | Dev 4 |
| `mlartifacts/` | artefatos dos runs, inclusive os modelos serializados; **nenhum modelo anterior é apagado** | Dev 4 |
| `models/modelo_vigente.json`, `models/historico_decisoes.csv` | manifesto do vigente e uma linha por decisão | Dev 3 |
| `data/processed/lotes.json` | lotes já ingeridos, hash de cada um e `data_version` | Dev 1 |
| `artifacts/<cycle_id>/` | features eleitas, tabela de votos, competição, decisão e previsões do ciclo | Devs 2, 3 e 4 |
| `evidencias/<cycle_id>/` | os 5 notebooks executados, `pipeline.log` e `resumo_ciclo.json` | Dev 4 |

As partições (`data/processed/*.parquet`) **não** são versionadas: são regeneradas a cada ciclo a partir
dos lotes, de forma determinística (hash do `Address`).

**Caminhos absolutos no `mlflow.db`.** O MLflow grava o caminho absoluto dos artefatos no banco (no runner,
`/home/runner/work/ml_ops_eth/ml_ops_eth/mlartifacts/...`). Para o mesmo banco funcionar em outra máquina,
`src.rastreio.relocalizar_artefatos()` reescreve a raiz antiga pela atual em todas as colunas de texto
(experimentos, runs, modelos logados e versões do registry). O `run_pipeline.sh` chama essa função antes
de cada ciclo, e `python -m src.rastreio ui` a chama antes de abrir a interface. A pasta de artefatos fica
sempre ao lado do `mlflow.db` em uso, então os testes, que usam um banco temporário, nunca sujam o
`mlartifacts/` do repositório.

### Falha não promove (critério 8)
- `run_pipeline.sh` usa `set -euo pipefail` e para no primeiro notebook que falhar.
- O manifesto `models/modelo_vigente.json` é o **último** arquivo escrito pelo notebook 04 (Dev 3), com
  escrita atômica (`.tmp` + `replace`).
- O commit do estado é um passo **posterior** ao pipeline. Se qualquer notebook falha, o job para antes do
  commit e o repositório continua exatamente no ciclo anterior. As evidências da falha ainda sobem como
  artifact do Actions, para depuração.
- O notebook 05 confere que o manifesto é do ciclo corrente e que o F1 de teste recalculado bate com o
  `f1_teste` gravado pelo 04. Qualquer divergência derruba o ciclo.

### Convenção de rastreio (critério 9)
Todo run do MLflow tem as tags `cycle_id`, `data_version` e `etapa`. Os runs disparados pelo Actions também
levam `gatilho` (`schedule` ou `workflow_dispatch`) e `github_run_id`. Com isso, qualquer número do
relatório volta à execução que o gerou:

```text
commit "🤖 Ciclo 20261117-1200-r2: promovido"
  └─ evidencias/20261117-1200-r2/resumo_ciclo.json ── github_run_url, data_version, decisão, run_ids
       └─ MLflow: tags.cycle_id = '20261117-1200-r2'  → 6 runs
            └─ models/modelo_vigente.json → run_id do vigente → mlartifacts/<run_id>/ (modelo)
```

### Previsões com o vigente (notebook 05)
O notebook 05 lê o manifesto, carrega o modelo **pelo `run_id`** (`runs:/<run_id>/modelo`) e prevê a
partição de **teste**, que nunca participou de nenhuma escolha. Como o modelo salvo é um `Pipeline` completo
(pré-processamento do Dev 1 → seleção das eleitas do Dev 2 → estimador do Dev 3), ele recebe as colunas do
parquet como estão. O notebook também mostra o que mudou entre os ciclos (`data_version`, features eleitas,
modelo vigente e decisão) e registra um run `previsoes` com `previsoes.csv` e a matriz de confusão.

## Evidências de execução

> **Simulação local (pendente de substituição).** Os números abaixo vieram de `run_pipeline.sh` executado
> num clone limpo do repositório, com `CYCLE_ID` fixo. O ciclo 1 foi commitado e o ciclo 2 rodou noutro
> caminho de disco, como acontece entre o runner e as máquinas do grupo. Esta seção deve ser atualizada com
> as execuções **agendadas** reais: id do run no Actions, gatilho `schedule` e link. Como o pipeline é
> determinístico, os números devem se repetir.

| Ciclo | Antes do ciclo | `data_version` | Features eleitas | F1 val A / B | Vigente anterior (F1 val reavaliado) | Decisão | Vigente após | F1 teste do vigente |
|---|---|---|---:|---|---|---|---|---:|
| 1 | só `lote_01` em `data/raw/` | `v1-65ada979` | 6 | 0,7036 / **0,9146** | — | **promovido** | B · XGBoost (ciclo 1) | 0,9171 |
| 2 | commit do `lote_02` | `v2-93f5fea1` | 7 | 0,7233 / **0,9400** | B do ciclo 1: 0,9064 | **promovido** (0,9400 ≥ 0,9064 + 0,01) | B · XGBoost (ciclo 2) | 0,9497 |
| 3 | nada (sem dados novos) | `v2-93f5fea1` | 7 | 0,7233 / 0,9400 | B do ciclo 2: 0,9400 | **mantido** (0,9400 < 0,9400 + 0,01) | B · XGBoost (ciclo 2) | 0,9497 |

- **Ciclo 2:** a feature `time_diff_between_first_and_last_mins` entrou na votação com o lote novo
  (6 → 7 eleitas). O vigente do ciclo 1 foi reavaliado na validação **do ciclo 2** (0,9064) antes da
  comparação, para que a régua fosse a mesma.
- **Ciclo 3:** mesmos lotes, mesmo `data_version`, treino idêntico (seeds fixas), então o desafiante empata
  com o vigente e não supera a margem de 0,01. A regra mantém o vigente, como esperado.
- **Previsões no teste (notebook 05):** ciclo 1, 1.367 carteiras (TP 293, FP 38, FN 15, TN 1.021);
  ciclos 2 e 3, 1.981 carteiras (TP 444, FP 32, FN 15, TN 1.490).
- **MLflow:** 6 runs por ciclo com o mesmo `cycle_id` (18 no total depois de 3 ciclos).
- **Duração:** cerca de 45 s de notebooks por ciclo, mais a instalação das dependências no runner.
- **Tamanho do estado:** cerca de 2,5 MB por ciclo (`mlflow.db` + `mlartifacts/` + `evidencias/`).

### Execuções agendadas (preencher)

| Ciclo | `cycle_id` | Gatilho | Run no Actions | Commit do estado | Decisão |
|---|---|---|---|---|---|
| 1 | | `schedule` | | | |
| 2 | | `schedule` | | | |
| 3 (opcional) | | `schedule` | | | |

Onde encontrar cada evidência:

1. **Actions → pipeline_mlops**, filtrando por *Event: schedule*: as execuções verdes, cada uma com um
   *job summary* (decisão, vigente, F1 e os 6 runs do MLflow).
2. **Commits** `🤖 Ciclo <cycle_id>: <decisão>` do `github-actions[bot]` na branch padrão.
3. **`evidencias/<cycle_id>/`**: os 5 notebooks executados com as saídas, o `pipeline.log` com o tempo de
   cada etapa e o `resumo_ciclo.json` com o link do run.
4. **MLflow** (`python -m src.rastreio ui`): filtro `tags.cycle_id = '<cycle_id>'` ou agrupamento pela
   coluna `cycle_id`; prints para o relatório.
5. **`models/historico_decisoes.csv`**: uma linha por ciclo, com a decisão e o motivo.
