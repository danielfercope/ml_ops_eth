# Dados — descrição, ETL e contrato (Dev 1)

> Base para a seção "Dados" do relatório N3 e contrato para as frentes 2–5.

## Fonte e licença
- **Dataset:** *Ethereum Fraud Detection Dataset*, Vagif Aliyev — Kaggle `vagifa/ethereum-frauddetection-dataset` (versão 1, 03/01/2021).
- **Arquivo:** `transaction_dataset.csv` — 9.841 linhas × 51 colunas (~2,9 MB).
- **Licença:** Open Database License (ODbL). Permite redistribuir com atribuição, então os lotes derivados ficam versionados no repositório. O CSV original é baixado com `kagglehub` (público, sem credenciais) e não vai para o git.

## Unidade de observação e alvo
- **Uma linha = uma carteira Ethereum** (`Address`), com estatísticas agregadas das suas transações normais e de tokens ERC20.
- **Alvo:** `FLAG` (1 = carteira fraudulenta, 0 = legítima). Fraude ≈ **22,1%** (2.179 de 9.841).
- **Métrica principal:** F1 da classe fraude (fixada em `config/config.yaml`).

## Problemas encontrados e tratamento
| Problema | Evidência | Tratamento |
|---|---|---|
| Espaços no início dos nomes | 26 colunas (ex.: `' Total ERC20 tnxs'`) | `strip` + snake_case |
| Identificadores como colunas | `Unnamed: 0`, `Index`, `Address` | removidos das features (`Address` só para dedup/partição) |
| `Address` duplicado | 25 no CSV completo, todos com o mesmo `FLAG` | mantida a 1ª ocorrência (lote antigo vence) |
| Colunas constantes | 7 colunas ERC20 (tempos médios e valores para contrato) | removidas — detectadas **só no treino** |
| Categóricas de alta cardinalidade | tipo de token mais enviado/recebido: 260–466 valores, com `"0"` e vazio como valores | descartadas |
| Ausentes | 829 linhas sem nenhum dado ERC20 (22 colunas juntas) | imputação pela mediana do treino |
| Caudas pesadas / outliers | assimetria até ~80; 15–20% fora de 1,5×IQR em várias colunas | `sign(x)·log1p(|x|)` + `StandardScaler` |

## Limitações (para o relatório)
- **A ausência de dados ERC20 só ocorre em fraudes** (38% das fraudes, 0% das legítimas). É artefato de coleta, não comportamento. Não criamos indicador de ausência para não dar um atalho espúrio ao modelo, mas a imputação ainda deixa um padrão detectável, então as métricas podem estar otimistas em relação a dados reais.
- Dados estáticos (snapshot de 2021): os "lotes" simulam a chegada de dados novos; não há deriva temporal real.
- Rótulos vêm de listas públicas de golpes; fraudes não reportadas aparecem como legítimas.
- Features agregadas por carteira: não há sequência temporal de transações.

## Lotes e versionamento
- `python -m src.ingestao gerar-lotes`: divide o CSV em **lote_01 (70%)** → `data/raw/` e **lote_02 (30%)** → `data/reserva/`, estratificado por `FLAG`, seed 42. Determinístico.
- **Antes do ciclo 2:** `git mv data/reserva/lote_02.csv data/raw/` e commit.
- `data_version = v<nº lotes>-<hash8>` (sha256 dos arquivos dos lotes, em ordem). Lote 1 → `v1-65ada979`.
- `data/processed/lotes.json` registra nome, sha256, linhas, fraudes, data e `cycle_id` de cada lote. Se um lote já registrado mudar de conteúdo, a ingestão dá erro (lotes são imutáveis).

## Partição
Hash MD5 de `"<sal>:<address em minúsculas>"` → número em [0,1): `< 0,60` treino, `< 0,80` validação, resto teste. O mesmo endereço cai sempre na mesma partição, então **adicionar o lote_02 não move nenhum registro antigo** (verificado em `tests/test_dados.py`). O teste só é usado no número final do relatório e no notebook 05.

Ciclo 1 (lote_01, 6.878 carteiras após dedup):

| Partição | Linhas | Fraudes | Taxa |
|---|---|---|---|
| treino | 4.205 | 946 | 22,5% |
| validação | 1.306 | 271 | 20,8% |
| teste | 1.367 | 308 | 22,5% |

## Contrato para as outras frentes
```python
from src.preprocessamento import carregar_particoes, carregar_schema, montar_preprocessador
treino, validacao, teste = carregar_particoes()      # DataFrames
features = carregar_schema()["features"]             # 38 nomes (ciclo 1)
pre = montar_preprocessador(features)                # NÃO ajustado: use dentro do Pipeline
X_tr, y_tr = treino[features], treino["flag"]
```
- Parquets: `data/processed/{treino,validacao,teste}.parquet` com `address`, `lote`, `flag` + features.
- `schema.json`: features, colunas descartadas e motivo, `data_version`, resumo das partições.
- O pré-processador devolve DataFrame com os mesmos nomes de coluna, então Dev 2 e Dev 3 podem selecionar as features eleitas por nome.

### Features (ciclo 1)
`avg_min_between_sent_tnx`, `avg_min_between_received_tnx`, `time_diff_between_first_and_last_mins`, `sent_tnx`, `received_tnx`, `number_of_created_contracts`, `unique_received_from_addresses`, `unique_sent_to_addresses`, `min_value_received`, `max_value_received`, `avg_val_received`, `min_val_sent`, `max_val_sent`, `avg_val_sent`, `min_value_sent_to_contract`, `max_val_sent_to_contract`, `avg_value_sent_to_contract`, `total_transactions_including_tnx_to_create_contract`, `total_ether_sent`, `total_ether_received`, `total_ether_sent_contracts`, `total_ether_balance`, `total_erc20_tnxs`, `erc20_total_ether_received`, `erc20_total_ether_sent`, `erc20_total_ether_sent_contract`, `erc20_uniq_sent_addr`, `erc20_uniq_rec_addr`, `erc20_uniq_sent_addr_1`, `erc20_uniq_rec_contract_addr`, `erc20_min_val_rec`, `erc20_max_val_rec`, `erc20_avg_val_rec`, `erc20_min_val_sent`, `erc20_max_val_sent`, `erc20_avg_val_sent`, `erc20_uniq_sent_token_name`, `erc20_uniq_rec_token_name` (38).
