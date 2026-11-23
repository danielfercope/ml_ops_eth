# Painel — contrato dos JSONs e mocks (Dev 5)

> Tarefa 1 do Dev 5 (critério 10): formato fechado dos cinco JSONs que o painel lê, mais versões mock
> para desenvolver as telas antes de existir modelo. Contrato com o Dev 4 (passo de exportação/deploy)
> e expectativa sobre o que o Dev 3 grava no manifesto.

## Onde fica cada coisa

| Caminho | O quê |
| --- | --- |
| `web/dados/schemas/*.schema.json` | **Fonte da verdade** do formato (JSON Schema 2020-12). `comum.schema.json` guarda os tipos compartilhados. |
| `web/dados/tipos.ts` | Os mesmos tipos em TypeScript, para as telas. Mudou um schema → muda aqui. |
| `web/` + `index.html` | App React (React Router, 5 telas). `web/dados/carregar.ts` busca os JSONs; `web/painel.test.tsx` testa as telas com os mocks (`npm test`). |
| `prototipo.html` | Protótipo aprovado em HTML único; origem do layout e dos design tokens (`web/estilos.css`). |
| `web/dados/mocks/<cenario>/*.json` | Mocks versionados (ver [Cenários](#cenários-mock)). |
| `src/contrato_painel.py` | Validador: schema + coerência entre os arquivos. O `exportar_painel.py` deve chamá-lo antes de publicar. |
| `tests/test_contrato_painel.py` | Valida todos os mocks, testa se o validador pega quebras e valida `public/data/` se existir. |
| `src/exportar_painel.py` | Gera os 5 JSONs a partir do estado do pipeline, valida com o `contrato_painel` e só então grava. |
| `tests/test_exportar_painel.py` | Exporta o estado do repositório e um 2º ciclo sintético ("mantido") e confere o contrato. |
| `public/data/*.json` | O que o painel busca em tempo de execução (`data/<nome>.json`, relativo à página). **Gerado pelo exportador e commitado pelo workflow** a cada ciclo. |

**Caminho de saída:** o PDF fala em `web/public/data/`, mas o Vite deste repo roda na raiz (o `index.html` e o
`public/` estão na raiz, e `web/` é só o código-fonte). Por isso a pasta servida é **`public/data/`**. O
Dev 4 deve apontar o passo de exportação para ela.

## Convenções (valem para os 5 arquivos)

- **Envelope** no topo de todo arquivo: `versao_formato` (hoje `1`), `gerado_em` e `origem` (`"mock"` ou
  `"pipeline"`). Com `origem = "mock"` a interface deve mostrar um aviso de dados fictícios.
- **Datas** em ISO 8601 UTC sem frações: `2026-11-17T09:06:12Z`. A hora do `cycle_id` também é UTC.
- **Métricas** são da classe fraude (`FLAG = 1`), entre 0 e 1, arredondadas a 4 casas. A formatação
  (`0,9429` ou `94,3%`) é responsabilidade da interface.
- **Ciclos em ordem cronológica** (mais antigo primeiro) em todo array `ciclos`. "Último ciclo" = último item.
- **`null` é explícito**: um campo opcional existe com `null`, nunca é omitido (assim o TypeScript não precisa
  tratar `undefined`).
- **Nada de campo extra**: os schemas usam `additionalProperties: false`. Campo novo = muda schema + tipos;
  se renomear/remover/mudar tipo, sobe `versao_formato`.
- **Modelos** são sempre identificados por `cycle_id` (ciclo em que foram treinados) + `modelo` (`A` ou `B`)
  - `algoritmo` + `run_id` do MLflow (`ref_modelo` no schema).

## Os cinco arquivos

| Arquivo | Tela | Conteúdo | Fonte para o exportador |
| --- | --- | --- | --- |
| `vigente.json` | Visão geral | modelo vigente, último ciclo, `modelo_mudou` (banner), modelo `anterior`, regra de promoção | `models/modelo_vigente.json` (Dev 3) + último item de `models/historico_decisoes.csv` + `config.yaml` |
| `ciclos.json` | Ciclos | por ciclo: data, gatilho, link da execução, `data_version`, lotes (e quais são novos), tamanhos das partições, vencedor, vigente antes/depois, decisão e motivo | `historico_decisoes.csv` (Dev 3) + `data/processed/lotes.json` e `schema.json` (Dev 1) + tags do MLflow (Dev 4) |
| `votacao.json` | Votação | regra, métodos e, por ciclo, a **tabela completa** feature × método (votos, ranks, rank médio, eleita, motivo) e o resumo | `artifacts/<cycle_id>/tabela_votos.csv`, `resumo_votacao.json` e `features_eleitas.json` (Dev 2) |
| `competicao.json` | Competição | por ciclo: precision/recall/F1 e matriz de confusão na validação de A, B e (do 2º ciclo em diante) do vigente reavaliado | runs do MLflow com tags `cycle_id` + `algoritmo` (Dev 3) |
| `previsoes.json` | Previsões | por ciclo: métricas e matriz no **teste inteiro** e uma amostra (address, lote, probabilidade, previsão, rótulo real) | saída do notebook 05 (Dev 4) |

Detalhes de cada campo estão nas `description` dos schemas e nos comentários de `tipos.ts`. Pontos que
merecem atenção:

- **`votacao.json` → `tabela`**: é o `tabela_votos.csv` do Dev 2 reorganizado. As colunas `voto_<metodo>` e
  `rank_<metodo>` viram os objetos `votos` e `ranks` (chave = `metodos[].id`), assim a tela não depende dos
  nomes dos métodos. `motivo` vazio no CSV vira `null`. A ordem das linhas é a mesma do CSV (eleitas primeiro).
- **`competicao.json` → `modelos`**: sempre `A`, `B` e, se havia vigente, `vigente`, nesta ordem. O vigente
  pode usar features diferentes das eleitas do ciclo (as do ciclo em que foi treinado), por isso cada modelo
  traz suas `features`.
- **`vigente_anterior.f1_val`** (em `ciclos.json`) e **`f1_vigente_anterior`** (em `vigente.json`) são o F1 do
  vigente **reavaliado na validação do ciclo atual**, que é o número que a regra de promoção compara.
- **`ciclos.json` → `qualidade`**: resultado das verificações de qualidade do vencedor (`aprovada`, `falhas`); `null` em
  ciclos decididos antes delas existirem. O validador exige `promovido` ⇔ limiar de F1 atendido **e** qualidade aprovada
  (ou `null`).
- **`vigente.json` → `modelo_mudou`** é `true` sempre que o último ciclo promoveu (inclusive o 1º ciclo, em
  que `anterior` é `null`).
- **`previsoes.json`**: as métricas usam todas as linhas do teste; a `amostra` é só para exibir (até 200
  linhas). O critério de amostragem vai em `criterio_amostra`.

## Regras de coerência (o que o validador confere além do schema)

Os mesmos `cycle_id`, na mesma ordem, em `ciclos`, `votacao` e `competicao`; `data_version` igual entre os
arquivos; `eleitas` = linhas com `eleita = true` = features de A e B; votos coerentes com `ranks` e `top_k`;
`rank_medio` = média dos ranks; ranks de cada método formam 1..N; vencedor = maior F1 entre A e B; decisão
compatível com `F1_val ≥ F1_vigente + 0,01` e `F1_val ≥ 0,60`; `vigente_anterior` de um ciclo =
`vigente_apos` do anterior; previsões feitas com o vigente do ciclo; `vigente.json` = último item de
`ciclos.json`; arquivos não misturam `origem` mock e pipeline.

## Cenários mock

| Cenário | Ciclos | Para testar |
| --- | --- | --- |
| `padrao` | c1 (`lote_01`) promove XGBoost; c2 (`+lote_02`) promove o novo XGBoost (0,9429 ≥ 0,9166 + 0,01) | banner de troca de modelo, comparação c1 × c2 na votação (entra `time_diff_between_first_and_last_mins`) |
| `mantido` | os dois acima + c3 sem dados novos: mesmo F1 do vigente, ganho 0 → **mantido** | último ciclo sem troca, `lotes_novos` vazio, linha "mantido" no histórico |
| `vazio` | nenhum | estado antes do 1º ciclo (`vigente: null`, listas vazias) |

**De onde vêm os números.** Rodei localmente o ETL e a votação do repositório sobre `lote_01` e depois
`lote_01 + lote_02`; os `data_version` (`v1-65ada979`, `v2-93f5fea1`), tamanhos das partições e tabelas de
votos batem com `docs/dados.md` e `docs/votacao_features.md`. Os modelos A e B foram treinados com
hiperparâmetros simples só para gerar métricas e probabilidades plausíveis. **São fictícios:** `cycle_id`,
datas, `run_id`, links do GitHub Actions e os `address` (hash dos reais, não correspondem a carteiras) — e as
métricas oficiais virão do Dev 3. O F1 alto (~0,93) é coerente com a limitação dos ERC20 ausentes descrita em
`docs/dados.md`.

## Como usar

```bash
python -m src.exportar_painel    # gera public/data/ a partir do estado do pipeline
npm run dev                      # o painel lê public/data/ (dados reais)
# http://localhost:5173/?cenario=padrao   → força um mock (padrao | mantido | vazio)
# sem public/data/, o painel cai sozinho no mock "padrao"

python -m src.contrato_painel                 # valida todos os cenários mock
python -m src.contrato_painel public/data     # valida o que o exportador gerou
python -m pytest -q tests/test_contrato_painel.py
```

## O que o exportador precisa dos outros devs

Para `src/exportar_painel.py` preencher tudo sem inventar campo:

| De quem | Preciso de |
| --- | --- |
| Dev 3 — `models/modelo_vigente.json` | `cycle_id`, `run_id`, algoritmo **e papel (A/B)**, `data_version`, `features`, `f1_val`, `f1_teste`, data da promoção |
| Dev 3 — `models/historico_decisoes.csv` (1 linha por ciclo) | `cycle_id`, data, papel + algoritmo + `run_id` + F1 do vencedor, `run_id` + F1 reavaliado do vigente anterior (vazio no 1º ciclo), decisão, motivo |
| Dev 3 — MLflow | run do vigente reavaliado na validação do ciclo, ou as métricas dele no histórico, para a terceira barra da Competição |
| Dev 4 — tags do MLflow | `gatilho` (`github.event_name`) e `execucao_url` (`$GITHUB_SERVER_URL/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID`); sem elas os campos saem `null` |
| Dev 4 — notebook 05 | grava `evidencias/<cycle_id>/previsoes.csv` com `address`, `lote`, `probabilidade`, `previsao`, `rotulo_real` do teste inteiro |
| Dev 4 — workflow | depois do notebook 05: `python -m src.exportar_painel --saida public/data` (já valida o contrato) → commit de `public/data/` → `npm ci && npm run build -- --base=/ml_ops_eth/` → deploy no Pages |
