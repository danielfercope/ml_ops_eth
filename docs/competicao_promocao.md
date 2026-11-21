# Competição de modelos e promoção (Dev 3)

> Base para as seções "Comparação dos modelos" e "Decisão de promoção" do relatório N3 (critérios 5 e 8).
> Código: `src/modelagem.py`, `src/promocao.py`. Notebooks: `notebooks/03_treino_competicao.ipynb`, `notebooks/04_selecao_promocao.ipynb`. Testes: `tests/test_modelagem.py`, `tests/test_promocao.py`.

## Objetivo
A cada ciclo, dois algoritmos competem **nas mesmas condições** (mesmas features eleitas, mesmo treino, mesma validação, seeds fixas). O vencedor do ciclo **só substitui o modelo vigente** se superar uma regra fixada antes do ciclo 1. A decisão é automática, fica registrada no MLflow e em `models/modelo_vigente.json`, e nenhum modelo anterior é apagado.

## Os candidatos
Cada candidato é um único `Pipeline` do scikit-learn, ajustado com um só `fit` no treino:

```
pre (Dev 1)                     selecao (Dev 2)                 modelo (Dev 3)
log com sinal → imputer         ColumnTransformer passthrough   LogisticRegression  (A)
(mediana) → StandardScaler  →   só as features eleitas,      →  ou
sobre as 38 candidatas          selecionadas por nome           XGBClassifier       (B)
```

Como pré-processamento e seleção estão dentro do pipeline, eles nunca veem validação nem teste, e o modelo salvo no MLflow já carrega tudo: o notebook 05 passa as colunas do parquet e recebe a probabilidade.

| | Modelo A | Modelo B |
|---|---|---|
| Algoritmo | `LogisticRegression` | `XGBClassifier` |
| Desbalanceamento (≈ 22% fraude) | `class_weight='balanced'` | `scale_pos_weight = legítimas / fraudes` do treino (≈ 3,45) |
| Hiperparâmetros | `C = 1`, `max_iter = 2000` | 300 árvores, `max_depth = 4`, `learning_rate = 0,05`, `subsample = colsample_bytree = 0,9` |
| Seed | 42 | 42 (`n_jobs = 1` para resultado idêntico entre máquinas) |

Os hiperparâmetros são **fixos** e estão em `config.yaml` → `modelos.hiperparametros`: a competição é entre algoritmos, não entre buscas de hiperparâmetro, e não há nenhum ajuste olhando a validação. A classe prevista usa o limiar 0,5 (`modelos.limiar_classificacao`).

## Comparação dos modelos

> **Simulação local.** Os números abaixo vieram dos notebooks 01 → 04 executados numa cópia do repositório, com `CYCLE_ID` fixo, exatamente como o workflow fará. Os números oficiais vêm das execuções agendadas (Dev 4) e esta seção deve ser atualizada com eles. Como o treino é determinístico, os valores devem se repetir.

### Ciclo 1: lote_01 (`data_version = v1-65ada979`, 6 features eleitas)
Treino: 4.205 carteiras (22,5% fraude). Validação: 1.306 (271 fraudes).

| Modelo | F1 | Precision | Recall | ROC AUC | Avg. precision | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A · Regressão Logística | 0,7036 | 0,5821 | 0,8893 | 0,9382 | 0,7927 | 241 | 173 | 30 | 862 |
| **B · XGBoost** | **0,9159** | **0,8889** | **0,9446** | **0,9938** | **0,9791** | 256 | 32 | 15 | 1.003 |

### Ciclo 2: lote_01 + lote_02 (`data_version = v2-93f5fea1`, 7 features eleitas)
Treino: 5.959 carteiras (22,4% fraude). Validação: 1.876 (384 fraudes). A votação do Dev 2 acrescentou `time_diff_between_first_and_last_mins`.

| Modelo | F1 | Precision | Recall | ROC AUC | Avg. precision | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A · Regressão Logística | 0,7233 | 0,6007 | 0,9089 | 0,9413 | 0,7803 | 349 | 232 | 35 | 1.260 |
| **B · XGBoost** | **0,9361** | **0,9196** | **0,9531** | **0,9969** | **0,9894** | 366 | 32 | 18 | 1.460 |

### Leitura
- **O XGBoost vence nos dois ciclos, com folga** (F1 +0,21). A diferença está quase toda na **precision**: os dois modelos encontram ~90–95% das fraudes, mas a Regressão Logística gera 5 a 7 vezes mais falsos positivos (173 contra 32 no ciclo 1).
- O motivo é a forma da fronteira. Com só 6–7 features agregadas, a separação entre fraude e legítima depende de **faixas e interações** (por exemplo, muitas origens distintas *e* valor médio atípico), que as árvores capturam e um modelo linear sobre as features transformadas não. O ROC AUC da LR ainda é alto (0,94): ela ordena razoavelmente, mas nenhum limiar linear isola bem as fraudes.
- Com o lote 2 (+42% de dados e 1 feature a mais) **os dois modelos melhoram**: A +0,020 e B +0,020 de F1. Para o XGBoost, os falsos positivos ficam estáveis (32) com 44% mais legítimas na validação, então a precision sobe de 0,889 para 0,920.
- A Regressão Logística continua útil como **linha de base interpretável**: se um dia o XGBoost estiver só marginalmente melhor, o desempate da regra prefere o modelo A.

Rastreio: um run MLflow por modelo (etapa `treino_competicao`) com tags `cycle_id`, `data_version`, `algoritmo`, `modelo`, `votacao_run_id`; params `hp.*`, `seed`, `features`; métricas `val_f1`, `val_precision`, `val_recall`, `val_roc_auc`, `val_tp/fp/fn/tn`; artefatos `modelo/`, `features.json`, matriz de confusão e figuras.

## Decisão de promoção

### Regra (fixada em `config.yaml` → `promocao` antes do ciclo 1)
1. **Vencedor do ciclo:** maior F1 de **validação** entre A e B. Empate → modelo A (mais simples).
2. **Vigente:** carregado do MLflow pelo `run_id` do manifesto e **reavaliado na validação do ciclo atual**, para que os dois sejam medidos com a mesma régua (a validação cresce com lotes novos).
3. **Promove** se `F1_val(vencedor) ≥ F1_val(vigente) + 0,01` **e** `F1_val(vencedor) ≥ 0,60`. Sem vigente (ciclo 1): basta `≥ 0,60`. Caso contrário, **mantém** o vigente.
4. **Teste:** só depois da decisão e só no modelo que ficou vigente, para o número do relatório. O teste nunca participa da escolha.

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `margem_minima` | 0,01 | trocar de modelo tem custo (revalidar, comunicar); diferenças menores que 1 ponto de F1 aparecem só por mudança de amostra |
| `f1_minimo` | 0,60 | piso absoluto: nenhum modelo abaixo disso vai para uso, nem no ciclo 1 |
| `desempate_vencedor` | A | em empate, o modelo mais simples e interpretável |

O teste `tests/test_promocao.py::test_regra_fixada_no_config` trava esses valores: mudar a regra depois de ver resultados quebra a suíte.

### Decisões dos ciclos (simulação local)

| Ciclo | data_version | Vencedor (F1_val) | Vigente antes (F1_val atual) | F1 necessário | Decisão | Vigente depois | F1 teste |
|---|---|---|---|---:|---|---|---:|
| 1 | v1-65ada979 | B · XGBoost (0,9159) | não há | 0,6000 | **promovido** | B do ciclo 1 | 0,9159 |
| 2 | v2-93f5fea1 | B · XGBoost (0,9361) | B do ciclo 1 (0,9095) | 0,9195 | **promovido** | B do ciclo 2 | 0,9497 |
| 3 (sem dados novos) | v2-93f5fea1 | B · XGBoost (0,9361) | B do ciclo 2 (0,9361) | 0,9461 | **mantido** | B do ciclo 2 | 0,9497 |

Motivos gravados automaticamente:
- **Ciclo 1:** "não há modelo vigente; o vencedor tem F1_val=0.9159 >= mínimo 0.60"
- **Ciclo 2:** "F1_val do vencedor 0.9361 >= F1_val do vigente 0.9095 + 0.01 e >= mínimo 0.60"
- **Ciclo 3:** "F1_val do vencedor 0.9361 < F1_val do vigente 0.9361 + margem 0.01 (precisava de 0.9461); o vigente foi mantido"

**Leitura.**
- No ciclo 2, o vigente do ciclo 1 **caiu de 0,9159 para 0,9095** ao ser medido na validação maior. Isso mostra por que ele é reavaliado a cada ciclo em vez de usar o F1 guardado: os números antigos não são comparáveis com os novos. O novo XGBoost, treinado com mais dados e com a feature de tempo de vida da carteira, superou o vigente por 2,7 pontos, acima da margem.
- O ciclo 3 roda sem dados novos: o pipeline determinístico reproduz exatamente o mesmo modelo (F1 0,9361), que não supera a si mesmo + 0,01. O resultado é **mantido**, que é o comportamento esperado da regra: sem evidência nova, nada muda.
- O F1 de teste do vigente (0,916 no ciclo 1 e 0,950 no ciclo 2) fica próximo do F1 de validação. É um sinal de que a escolha pela validação não superestimou o modelo.

### Artefatos da decisão

**`models/modelo_vigente.json`** (manifesto, consumido pelo notebook 05 e pelo painel), reescrito a cada ciclo:

| Campo | Conteúdo |
|---|---|
| `cycle_id`, `data_version` | ciclo da **última decisão** |
| `decisao`, `motivo`, `f1_necessario`, `decidido_em` | resultado da regra |
| `run_id`, `model_uri`, `modelo`, `algoritmo` | modelo vigente após a decisão (`model_uri = runs:/<run_id>/modelo`) |
| `features`, `n_features` | features que o vigente usa (as do ciclo em que foi treinado) |
| `f1_validacao`, `f1_teste`, `metricas_validacao`, `metricas_teste` | vigente medido na validação e no teste **do ciclo atual** |
| `treinado_no_ciclo`, `treinado_com_data_version`, `promovido_no_ciclo`, `promovido_em` | origem do vigente |
| `desafiante`, `candidatos`, `vigente_anterior` | contexto da comparação (run_id e F1 de cada um) |
| `regra`, `decisao_run_id`, `registry` | regra aplicada, run da decisão no MLflow, versão no Model Registry |

**`models/historico_decisoes.csv`**: uma linha por ciclo (rodar o mesmo `cycle_id` de novo substitui a linha em vez de duplicar), com `f1_val_A`, `f1_val_B`, vencedor, vigente anterior, F1 necessário, decisão, motivo, vigente resultante, F1 de validação e de teste.

**`artifacts/<cycle_id>/`**: `competicao.json` e `comparacao_modelos.csv` (notebook 03) e `decisao.json` (cópia do manifesto daquele ciclo).

**MLflow**: um run `promocao_<cycle_id>` (etapa `promocao`) com tag `decisao`, métricas `promovido` (0/1), `f1_val_A`, `f1_val_B`, `f1_val_vencedor`, `f1_val_vigente_anterior`, `f1_necessario`, `f1_val_vigente`, `f1_teste_vigente` e o artefato `decisao.json`. Quando há promoção, o modelo também é registrado no **Model Registry** (`fraude_ethereum`) e o alias `vigente` passa a apontar para a nova versão. Esse registro é só um espelho para a interface do MLflow: a fonte da verdade continua sendo o manifesto.

### Proteções contra promoção indevida
- O manifesto é **o último arquivo escrito** no notebook 04, com escrita atômica (arquivo temporário + `rename`). Se qualquer célula falhar antes, o vigente não muda e o workflow não commita nada.
- O notebook 03 só aceita features eleitas sobre o `data_version` atual (`resolver_ciclo`); o 04 confere o `data_version` da competição.
- Se o vigente não puder ser avaliado nas partições atuais (por exemplo, se uma coluna dele sumir do schema), o notebook para com erro em vez de promover às cegas.
- Nenhum run é apagado: o manifesto só troca o ponteiro, e versões antigas continuam no MLflow e no Registry.

## Limitações
- Hiperparâmetros fixos: um XGBoost ajustado (ou a LR com interações) poderia mudar o placar, mas buscar hiperparâmetros na validação exigiria outra partição para a decisão de promoção.
- O F1 é medido no limiar 0,5. Em produção, o limiar seria escolhido pelo custo relativo de falso positivo e falso negativo.
- Os dados ERC20 ausentes aparecem só em fraudes (ver `docs/dados.md`). As árvores exploram esse padrão com mais facilidade que a LR, então parte da vantagem do XGBoost pode ser artefato de coleta e as métricas podem estar otimistas.
- Com uma única validação por ciclo, a margem de 0,01 é uma regra prática, não um teste estatístico.

## Como reproduzir
```bash
python -m src.preprocessamento                                                  # Dev 1
CYCLE_ID=<id> jupyter nbconvert --to notebook --execute notebooks/02_votacao_features.ipynb
CYCLE_ID=<id> jupyter nbconvert --to notebook --execute notebooks/03_treino_competicao.ipynb
CYCLE_ID=<id> jupyter nbconvert --to notebook --execute notebooks/04_selecao_promocao.ipynb
python -m pytest -q tests                                                       # inclui Dev 1 e 2
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Carregar o vigente (notebook 05):
```python
import json, mlflow, mlflow.sklearn
from src.preprocessamento import carregar_particoes
mlflow.set_tracking_uri("sqlite:///mlflow.db")              # ou src.features.configurar_mlflow()
man = json.load(open("models/modelo_vigente.json"))
modelo = mlflow.sklearn.load_model(man["model_uri"])        # pipeline completo
_, _, teste = carregar_particoes()
proba = modelo.predict_proba(teste)[:, 1]                   # passa o parquet direto
```
Para simular os ciclos localmente sem sujar o repositório, rode numa cópia: o `models/modelo_vigente.json` local faria o ciclo 1 oficial encontrar um vigente que não existe no MLflow do workflow.
