# Recurso adicional: painel de acompanhamento (Dev 5)

> Base para a seção "Recurso adicional" do relatório N3 (critério 10).
> Código: `web/` + `index.html` (app), `prototipo.html` (protótipo aprovado), `src/exportar_painel.py`,
> `src/contrato_painel.py`. Testes: `web/painel.test.tsx` (Vitest), `tests/test_exportar_painel.py` e
> `tests/test_contrato_painel.py` (pytest). Contrato dos dados: [`painel_dados.md`](painel_dados.md).

## Problema

O pipeline grava tudo o que acontece em cada ciclo, mas espalhado: runs e métricas no `mlflow.db`, tabela de
votos e competição em `artifacts/<cycle_id>/`, a decisão em `models/modelo_vigente.json` e no
`historico_decisoes.csv`, as previsões no notebook 05 executado em `evidencias/<cycle_id>/`. Para responder
perguntas simples ("qual modelo está em produção?", "o modelo mudou no último ciclo? por quê?", "quais
features entraram com o lote novo?") era preciso rodar `mlflow ui` localmente ou abrir notebooks e CSVs.

## Justificativa

- **Visibilidade da decisão automática.** A promoção acontece sem intervenção humana; o painel mostra a
  regra, os números comparados (F1 do vencedor × vigente reavaliado × exigido) e o motivo, em uma tela.
- **Evolução entre ciclos.** O MLflow lista runs; o painel compara ciclos: votação do ciclo N contra o
  N−1 (features que entraram e saíram), histórico de decisões e tamanho dos dados a cada lote.
- **Acesso sem instalação.** Publicado no GitHub Pages, abre no navegador, sem servidor, API ou
  credenciais, o que serve ao grupo e à avaliação.

## Solução

Um app **Vite + React + TypeScript** com **React Router** e cinco telas, alimentado por cinco JSONs estáticos:

| Tela | Responde | Arquivo |
| --- | --- | --- |
| Visão geral | modelo vigente (algoritmo, `cycle_id`, `data_version`, F1), última decisão e banner quando o modelo mudou | `vigente.json` |
| Ciclos | histórico com data, gatilho, `data_version`, lotes, vencedor, F1 e decisão de cada ciclo | `ciclos.json` |
| Votação | tabela feature × método do ciclo escolhido, eleitas destacadas, diferença para o ciclo anterior | `votacao.json` |
| Competição | barras agrupadas de precision, recall e F1 de A, B e vigente reavaliado, e matrizes de confusão | `competicao.json` |
| Previsões | amostra do teste com probabilidade, previsão e rótulo real, métricas e matriz do teste inteiro | `previsoes.json` |

Decisões de projeto:

- **Contrato antes das telas.** O formato dos JSONs foi fechado primeiro (JSON Schemas em
  `web/dados/schemas/` + tipos em `web/dados/tipos.ts`), com mocks de três cenários (`padrao`, `mantido`,
  `vazio`) para desenvolver as telas antes de existir modelo. `src/contrato_painel.py` valida schema e
  coerência entre os arquivos (mesmos `cycle_id`, eleitas = features de A e B, decisão compatível com a
  regra de promoção etc.).
- **Protótipo aprovado → app.** As telas foram primeiro prototipadas em HTML único (`prototipo.html`),
  aprovadas pelo grupo, e então portadas para React reaproveitando os design tokens (`web/estilos.css`).
- **Sem backend.** O painel é estático: o exportador gera os JSONs e o Pages serve os arquivos. As rotas usam
  `HashRouter` (`#/ciclos`), que funciona no Pages sem configuração de servidor.

## Ponto de integração

O painel entra no **mesmo workflow** (`pipeline_mlops.yml`), depois do notebook 05:

1. Job `ciclo`: `python -m src.exportar_painel --saida public/data` lê o estado do pipeline e **valida o
   contrato antes de gravar**; se algo não bater, nada é escrito. `public/data/` é commitado junto com o
   estado do ciclo (`🤖 Ciclo <cycle_id>: <decisão>`).
2. Job `painel`: faz checkout do commit do ciclo, roda `npm ci`, `npm test` (Vitest) e
   `npm run build -- --base=/ml_ops_eth/` e publica `dist/` no GitHub Pages.

Os dois passos usam `continue-on-error`: uma falha no painel nunca deixa o ciclo vermelho nem impede a
promoção ou a persistência do MLflow. O painel é atualizado a cada ciclo sem intervenção manual.

## Resultado

- **Ciclo 1** (`20261121-1311-r1`, `data_version` `v1-65ada979`): o painel mostra o XGBoost (Modelo B)
  promovido como primeiro vigente, com F1 de validação 0,9159 contra o mínimo de 0,60 (Regressão Logística:
  0,7036), 6 features eleitas e, no teste, F1 0,9159 (VP 294, FP 40, FN 14, VN 1.019).
- **Ciclo 2:** depende da entrada do `lote_02` e da execução agendada. Quando rodar, o painel passa a mostrar a
  terceira barra (vigente reavaliado), a comparação da votação com o ciclo 1 e o banner de troca, se houver
  promoção.

**Prints para o relatório:** abrir o painel publicado (ou `npm run dev`) nas telas Visão geral, Votação e
Competição depois dos ciclos 1 e 2. `?tema=light` força o tema claro para impressão.
