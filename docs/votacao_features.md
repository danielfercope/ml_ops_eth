# Votação das features — três métodos e eleição por ciclo (Dev 2)

> Base para a seção "Votação das features" do relatório N3 (critérios 3 e 4).
> Código: `src/features.py`; notebook: `notebooks/02_votacao_features.ipynb`; testes: `tests/test_features.py`, `tests/test_regra_votacao.py`, `tests/test_votacao.py`, `tests/test_registro_mlflow.py`.

## Objetivo
Em vez de reduzir dimensionalidade com PCA (o que destruiria a interpretabilidade das features), cada ciclo elege um **subconjunto das 38 features** do Dev 1 por **votação entre três métodos de seleção**. O conjunto eleito é o mesmo para os dois modelos do Dev 3, que selecionam as colunas por nome na saída do pré-processador.

## Os três métodos
Todos recebem o treino já pré-processado (log com sinal → imputação pela mediana → padronização, ajustado **só no treino**) e devolvem o ranking completo das 38 features mais as top-k.

| Método (nome no config) | O que mede | Enxerga | Parâmetros |
|---|---|---|---|
| `selectkbest_f_classif` | estatística F (ANOVA) de cada feature isolada contra `flag` | relações univariadas | `k = 10` |
| `random_forest` | redução média de impureza (Gini) | não linearidades e interações | 300 árvores, `class_weight='balanced'`, seed 42 |
| `rfe_logistic` | ordem de eliminação recursiva com Regressão Logística balanceada | contribuição conjunta num modelo linear | passo 1, elimina até sobrar 1 (dá a ordem completa), `max_iter = 2000`, seed 42 |

Validação e teste **não** são usados: a seleção é parte do treinamento e, se olhasse para a validação, as métricas do Dev 3 ficariam otimistas (critério 2).

## Regra de votação (fixada antes do ciclo 1)
Gravada em `config/config.yaml` → `votacao` e lida por `carregar_regra_votacao()`, o único ponto de leitura. O teste `tests/test_regra_votacao.py` trava os valores: alterar a regra depois de ver resultados quebra a suíte e aparece na revisão.

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `top_k` | 10 por método | ≈ 26% das 38 candidatas: grande o bastante para os métodos se sobreporem, pequeno o bastante para cada voto significar algo |
| `votos_minimos` | 2 de 3 | maioria simples: uma feature só entra se dois critérios de natureza diferente (estatístico, de árvore, linear) concordarem, o que filtra ruído valorizado por um método isolado |
| `desempate` | média dos ranks | usa a informação completa dos três rankings, não só quem está no top-k; determinística |
| `minimo_features` | 5 | se os métodos discordarem muito, os modelos ainda recebem um conjunto utilizável; as completadas ficam marcadas com `motivo = completou_minimo` |

## Ciclo 1 — lote_01 (`data_version = v1-65ada979`)
Treino: 4.205 carteiras (22,5% fraude). Top-10 de cada método:

| rank | SelectKBest (F) | Random Forest | RFE (Logística) |
|---|---|---|---|
| 1 | total_ether_received | total_erc20_tnxs | erc20_total_ether_received |
| 2 | avg_val_received | erc20_uniq_rec_contract_addr | erc20_avg_val_rec |
| 3 | max_value_received | erc20_uniq_rec_addr | erc20_uniq_rec_token_name |
| 4 | total_ether_sent | time_diff_between_first_and_last_mins | erc20_max_val_rec |
| 5 | max_val_sent | erc20_uniq_rec_token_name | total_transactions_including_tnx_to_create_contract |
| 6 | sent_tnx | avg_min_between_received_tnx | unique_received_from_addresses |
| 7 | total_transactions_including_tnx_to_create_contract | total_ether_received | total_erc20_tnxs |
| 8 | min_value_received | total_transactions_including_tnx_to_create_contract | erc20_uniq_sent_addr |
| 9 | avg_val_sent | avg_val_received | erc20_max_val_sent |
| 10 | erc20_min_val_rec | unique_received_from_addresses | erc20_avg_val_sent |

**Concordância entre os top-10** (features em comum): RF ∩ RFE = 4, RF ∩ KBest = 3, KBest ∩ RFE = 1. Os métodos olham para ângulos diferentes: o F-test privilegia volumes de ETH recebidos/enviados, a RFE privilegia as colunas ERC20 e a floresta mistura os dois grupos com contagens e tempo. É exatamente o cenário em que a votação faz sentido.

**Tabela de votos** (features com ao menos 1 voto; as 15 restantes não receberam voto):

| Feature | KBest | RF | RFE | Votos | Rank médio | Eleita |
|---|:-:|:-:|:-:|:-:|---:|:-:|
| total_transactions_including_tnx_to_create_contract | 1 | 1 | 1 | **3** | 6,67 | ✔ |
| avg_val_received | 1 | 1 | 0 | **2** | 7,67 | ✔ |
| total_ether_received | 1 | 1 | 0 | **2** | 9,67 | ✔ |
| erc20_uniq_rec_token_name | 0 | 1 | 1 | **2** | 11,33 | ✔ |
| total_erc20_tnxs | 0 | 1 | 1 | **2** | 15,33 | ✔ |
| unique_received_from_addresses | 0 | 1 | 1 | **2** | 15,67 | ✔ |
| max_value_received | 1 | 0 | 0 | 1 | 10,33 | |
| erc20_total_ether_received | 0 | 0 | 1 | 1 | 12,33 | |
| time_diff_between_first_and_last_mins | 0 | 1 | 0 | 1 | 12,67 | |
| sent_tnx | 1 | 0 | 0 | 1 | 13,67 | |
| erc20_uniq_rec_contract_addr | 0 | 1 | 0 | 1 | 14,33 | |
| erc20_max_val_rec | 0 | 0 | 1 | 1 | 14,67 | |
| min_value_received | 1 | 0 | 0 | 1 | 16,00 | |
| total_ether_sent | 1 | 0 | 0 | 1 | 16,00 | |
| erc20_min_val_rec | 1 | 0 | 0 | 1 | 16,33 | |
| erc20_uniq_sent_addr | 0 | 0 | 1 | 1 | 16,67 | |
| erc20_avg_val_rec | 0 | 0 | 1 | 1 | 17,33 | |
| erc20_uniq_rec_addr | 0 | 1 | 0 | 1 | 18,33 | |
| erc20_max_val_sent | 0 | 0 | 1 | 1 | 18,33 | |
| avg_min_between_received_tnx | 0 | 1 | 0 | 1 | 19,67 | |
| erc20_avg_val_sent | 0 | 0 | 1 | 1 | 20,00 | |
| max_val_sent | 1 | 0 | 0 | 1 | 20,00 | |
| avg_val_sent | 1 | 0 | 0 | 1 | 20,33 | |

**Resultado do ciclo 1:** 6 features eleitas (1 com 3 votos, 5 com 2 votos), todas por votos; o mínimo de 5 não precisou ser acionado. 17 features ficaram com 1 voto e 15 sem voto.

O conjunto eleito é interpretável: volume e valor médio recebidos em ETH (`total_ether_received`, `avg_val_received`), diversidade de origem (`unique_received_from_addresses`), atividade total (`total_transactions_including_tnx_to_create_contract`) e atividade em tokens ERC20 (`total_erc20_tnxs`, `erc20_uniq_rec_token_name`). Carteiras fraudulentas tendem a receber de muitas origens, em valores atípicos, e a interagir com tokens de forma diferente das legítimas.

## Ciclo 2 — lote_01 + lote_02 (`data_version = v2-93f5fea1`, simulação local)
> Simulado copiando `data/reserva/lote_02.csv` para `data/raw/` e rodando o ETL e o notebook 02 localmente, sem commit. Os números oficiais do ciclo 2 vêm da execução agendada do workflow (Dev 4), que grava `artifacts/<cycle_id>/`; esta seção deve ser atualizada com eles.

Treino: 5.959 carteiras (22,4% fraude); 25 endereços duplicados removidos entre os lotes; nenhum registro antigo mudou de partição. As mesmas 38 features candidatas (as 7 constantes do ciclo 1 continuam constantes).

**Comparação das votações** (features com ao menos 1 voto em algum ciclo):

| Feature | Votos c1 | Votos c2 | Rank médio c1 | Rank médio c2 | c1 | c2 |
|---|:-:|:-:|---:|---:|:-:|:-:|
| total_transactions_including_tnx_to_create_contract | 3 | 3 | 6,67 | 6,33 | ✔ | ✔ |
| avg_val_received | 2 | 2 | 7,67 | 7,67 | ✔ | ✔ |
| total_ether_received | 2 | 2 | 9,67 | 7,67 | ✔ | ✔ |
| **time_diff_between_first_and_last_mins** | 1 | **2** | 12,67 | 10,67 | | **✔ (entrou)** |
| erc20_uniq_rec_token_name | 2 | 2 | 11,33 | 11,33 | ✔ | ✔ |
| total_erc20_tnxs | 2 | 2 | 15,33 | 15,33 | ✔ | ✔ |
| unique_received_from_addresses | 2 | 2 | 15,67 | 16,00 | ✔ | ✔ |
| max_value_received | 1 | 1 | 10,33 | 10,67 | | |
| erc20_total_ether_received | 1 | 1 | 12,33 | 13,00 | | |
| sent_tnx | 1 | 1 | 13,67 | 13,33 | | |
| erc20_max_val_rec | 1 | 1 | 14,67 | 14,00 | | |
| erc20_uniq_rec_contract_addr | 1 | 1 | 14,33 | 14,67 | | |
| total_ether_sent | 1 | 1 | 16,00 | 15,33 | | |
| erc20_avg_val_rec | 1 | 1 | 17,33 | 17,00 | | |
| min_value_received | 1 | 1 | 16,00 | 17,00 | | |
| erc20_uniq_rec_addr | 1 | 1 | 18,33 | 17,33 | | |
| erc20_uniq_sent_addr | 1 | 1 | 16,67 | 17,33 | | |
| erc20_max_val_sent | 1 | 1 | 18,33 | 17,67 | | |
| avg_min_between_received_tnx | 1 | 1 | 19,67 | 18,00 | | |
| erc20_avg_val_sent | 1 | 1 | 20,00 | 18,67 | | |
| max_val_sent | 1 | 1 | 20,00 | 19,67 | | |
| avg_val_sent | 1 | 1 | 20,33 | 20,00 | | |
| erc20_min_val_rec | 1 | **0** | 16,33 | 17,33 | | |

**O que mudou com o lote novo:**
- **As 6 eleitas do ciclo 1 foram mantidas**, com votos idênticos: o conjunto é estável quando os dados crescem ~42%.
- **Entrou `time_diff_between_first_and_last_mins`** (tempo de vida da carteira): no ciclo 1 tinha só o voto da floresta (rank 4 na RF, 11 no F-test, ou seja, a 1ª fora do top-10); com mais dados subiu para rank 2 na RF e **rank 10 no F-test**, ganhando o 2º voto. A RFE continua não a valorizando (rank 20).
- **`erc20_min_val_rec` perdeu o único voto** que tinha (rank 10 → 11 no F-test), cedendo a vaga justamente para `time_diff`.
- Concordância RF ∩ KBest subiu de 3 para 4; as outras duas não mudaram. A RFE segue como o método mais "divergente" (1 feature em comum com o F-test).
- Resultado: **7 eleitas** (1 com 3 votos, 6 com 2), nenhuma completada pelo mínimo. O Dev 3 treina o ciclo 2 com uma feature a mais do que o ciclo 1, e o `features_eleitas.json` de cada ciclo registra isso.

A mudança é pequena e explicável, o que é o comportamento desejado: a votação refaz a escolha a cada ciclo (critério 4), mas só muda o conjunto quando há evidência nos dados, não por ruído de um único método.

## Rastreabilidade (MLflow e artefatos)
Por ciclo, um run na etapa `votacao_features`, no mesmo experimento e `mlflow.db` do notebook 01:

| Onde | O quê |
|---|---|
| tags | `cycle_id`, `data_version`, `etapa = votacao_features` |
| params | `regra.*` (top_k, votos_minimos, desempate, minimo_features, metodos); `<metodo>.*` (hiperparâmetros de cada método); `features_eleitas` |
| metrics | `n_eleitas`, `n_eleitas_por_votos`, `n_completadas_pelo_minimo`, `n_com_3_votos`, `n_com_2_votos`, `n_com_1_voto`, `n_sem_votos` |
| artifacts | `rankings/<metodo>.csv`, `tabela_votos.csv`, `features_eleitas.json`, `resumo_votacao.json`, `figuras/*.png` |

Os mesmos arquivos ficam em `artifacts/<cycle_id>/` no repositório. **Contrato com o Dev 3:** `carregar_features_eleitas(cycle_id)["features"]` (lista ordenada); o JSON traz também `data_version`, `mlflow_run_id`, a `regra`, o `motivo` de cada eleita e o `top_k_por_metodo`. **Contrato com o Dev 5:** `tabela_votos.csv` com `feature`, `voto_<metodo>`, `total_votos`, `rank_<metodo>`, `rank_medio`, `eleita`, `motivo`.

## Limitações
- Os três métodos são ajustados no treino inteiro, sem validação cruzada interna; com 4–6 mil linhas a variância dos rankings é baixa (os dois ciclos mostram isso), mas um ciclo com lote pequeno poderia oscilar mais perto do corte do top-10.
- O F-test assume relação linear/univariada e a RFE assume um modelo linear; não linearidades só são capturadas pela floresta, que tem 1 voto. Features puramente de interação precisariam de 2 métodos não lineares para serem eleitas.
- A regra elege por maioria, não por desempenho: um conjunto com mais features poderia dar F1 maior. A escolha deliberada foi por estabilidade e interpretabilidade, e o Dev 3 mede o efeito na validação.
- O ciclo 2 desta seção é uma simulação local; a evidência oficial é o `artifacts/<cycle_id>/` gerado pelo workflow agendado.

## Como reproduzir
```bash
python -m src.preprocessamento                       # parquets e schema.json do ciclo atual (Dev 1)
CYCLE_ID=<id> jupyter nbconvert --to notebook --execute notebooks/02_votacao_features.ipynb
python -m pytest -q tests                            # inclui os testes do Dev 1
mlflow ui --backend-store-uri sqlite:///mlflow.db    # run votacao_features_<id>
```
Para simular o ciclo 2 localmente: copiar `data/reserva/lote_02.csv` para `data/raw/` e repetir os dois primeiros comandos (sem commitar o lote nem o `mlflow.db`).
