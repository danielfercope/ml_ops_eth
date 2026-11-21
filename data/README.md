# Dados — fonte e licença

Os arquivos em `data/raw/` e `data/reserva/` são derivados do
**Ethereum Fraud Detection Dataset**, de **Vagif Aliyev**, publicado no Kaggle:
https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset

O dataset original é disponibilizado sob a **Open Database License (ODbL) v1.0**
(https://opendatacommons.org/licenses/odbl/1-0/). Em cumprimento à licença:

- **Atribuição:** os créditos do dataset original pertencem ao autor citado acima.
- **Compartilhamento pela mesma licença:** os lotes derivados (`lote_01.csv`, `lote_02.csv`)
  continuam sob a ODbL v1.0.
- **Modificações feitas:** o CSV original foi apenas dividido em dois lotes (70% / 30%),
  estratificados pela coluna `FLAG`, com seed fixa (`python -m src.ingestao gerar-lotes`).
  Nenhum valor foi alterado.

| Pasta | Conteúdo | Versionado no git? |
|---|---|---|
| `data/original/` | CSV original baixado do Kaggle | não |
| `data/raw/` | lotes que o pipeline usa no ciclo atual | sim |
| `data/reserva/` | `lote_02`, que vai para `data/raw/` antes do ciclo 2 | sim |
| `data/processed/` | partições e `schema.json` (gerados a cada ciclo); `lotes.json` é commitado pelo workflow | só `lotes.json` |
