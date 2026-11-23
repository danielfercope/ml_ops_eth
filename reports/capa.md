# Relatório técnico — Trabalho N3 de MLOps

**Detecção de fraude em carteiras Ethereum com votação de features, competição de modelos e retreinamento agendado**

Católica de Santa Catarina · Análise Preditiva · Engenharia de Software e Engenharia de IA

**Integrantes:** _preencher com os nomes do grupo_

**Repositório:** <https://github.com/danielfercope/ml_ops_eth> · **Painel:** <https://danielfercope.github.io/ml_ops_eth/>

## Visão geral

Um único workflow agendado no GitHub Actions (`.github/workflows/pipeline_mlops.yml`) executa, a cada ciclo, o pipeline
inteiro: incorpora lotes do *Ethereum Fraud Detection Dataset* (Kaggle), faz o ETL sem vazamento, elege as features por
votação de três métodos de seleção, treina dois algoritmos com as mesmas features (A = Regressão Logística,
B = XGBoost), registra tudo no MLflow e só troca o modelo vigente se o vencedor superar um limiar fixado antes do
primeiro ciclo e passar nas verificações de qualidade. O modelo vigente faz as previsões do ciclo e um painel publicado
no GitHub Pages mostra a evolução. Cada ciclo é identificado por `cycle_id` e `data_version`.

## Mapa dos critérios de avaliação

| Critério | Onde está no relatório | Evidência no repositório |
|---|---|---|
| 1. Problema, dados e avaliação | Dados: fonte, licença, alvo, métrica e partição | `config/config.yaml`, `notebooks/01_etl.ipynb` |
| 2. ETL e engenharia de atributos | Dados: ETL, leakage e versionamento | `src/ingestao.py`, `src/preprocessamento.py`, `data/processed/lotes.json` |
| 3. Três métodos de seleção | Votação das features: os três métodos | `src/features.py`, `notebooks/02_votacao_features.ipynb` |
| 4. Votação e eleição | Votação das features: regra e tabela de votos | `artifacts/<cycle_id>/tabela_votos.csv`, `features_eleitas.json` |
| 5. Treino dos dois algoritmos | Competição: candidatos e comparação | `src/modelagem.py`, `notebooks/03_treino_competicao.ipynb` |
| 6. MLflow e rastreabilidade | Arquitetura: MLflow e persistência entre ciclos | `mlflow.db`, `mlartifacts/`, tags `cycle_id`/`data_version` |
| 7. Workflow único e agendamento | Arquitetura e evidências de execução | `pipeline_mlops.yml`, execuções com gatilho `schedule` |
| 8. Eleição, promoção e previsões | Decisão de promoção, verificações de qualidade, previsões | `src/promocao.py`, `models/modelo_vigente.json`, `notebooks/05_previsoes.ipynb` |
| 9. Repositório, relatório e evidências | Este relatório e as evidências de execução | `README.md`, `evidencias/<cycle_id>/` |
| 10. Recurso adicional | Recurso adicional: painel de acompanhamento | `web/`, `src/exportar_painel.py`, GitHub Pages |
