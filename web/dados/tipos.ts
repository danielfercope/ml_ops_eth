/**
 * Contrato dos JSONs do painel (Dev 5, critério 10) — lado TypeScript.
 *
 * Fonte da verdade: os JSON Schemas em `web/dados/schemas/`. Estes tipos os
 * espelham; ao mudar um schema, mude aqui também e suba `versao_formato`
 * se a mudança quebrar o formato. Documentação: `docs/painel_dados.md`.
 *
 * Arquivos servidos em `${import.meta.env.BASE_URL}data/<nome>.json`.
 */

// --------------------------------------------------------------------------
// Tipos comuns
// --------------------------------------------------------------------------

/** ISO 8601 em UTC, ex.: "2026-11-17T09:06:12Z". */
export type DataHora = string
/** "AAAAMMDD-HHMM-<sufixo>", ex.: "20261117-0900-r15". */
export type CycleId = string
/** "v<nº lotes>-<hash8>", ex.: "v2-93f5fea1". */
export type DataVersion = string
/** run_id do MLflow (32 hex). */
export type RunId = string
/** "lote_01", "lote_02", ... */
export type Lote = string

/** A = Regressão Logística, B = XGBoost. */
export type Papel = 'A' | 'B'
export type Algoritmo = 'logistic_regression' | 'xgboost'
export type Decisao = 'promovido' | 'mantido'
export type Gatilho = 'schedule' | 'workflow_dispatch' | 'local'
export type Metrica = 'precision' | 'recall' | 'f1'

export interface Envelope {
  versao_formato: 1
  gerado_em: DataHora
  /** "mock" = dados fictícios (mostrar aviso na interface). */
  origem: 'mock' | 'pipeline'
}

/** Métricas da classe fraude, entre 0 e 1, arredondadas a 4 casas. */
export type Metricas = Record<Metrica, number>

export interface MatrizConfusao {
  vn: number
  fp: number
  fn: number
  vp: number
}

/** Aponta para um modelo treinado (ciclo em que foi treinado + run do MLflow). */
export interface RefModelo {
  cycle_id: CycleId
  modelo: Papel
  algoritmo: Algoritmo
  run_id: RunId
}

// --------------------------------------------------------------------------
// ciclos.json — tela Ciclos
// --------------------------------------------------------------------------

export interface Ciclo {
  cycle_id: CycleId
  executado_em: DataHora
  gatilho: Gatilho | null
  execucao_url: string | null
  data_version: DataVersion
  lotes: Lote[]
  /** Vazio = ciclo sem dados novos. */
  lotes_novos: Lote[]
  dados: {
    n_carteiras: number
    duplicados_removidos: number
    n_treino: number
    n_validacao: number
    n_teste: number
    taxa_fraude_treino: number
  }
  n_features_eleitas: number
  vencedor: RefVencedor
  /** Vigente no início do ciclo, F1 reavaliado na validação deste ciclo. null no 1º ciclo. */
  vigente_anterior: (RefModelo & { f1_val: number }) | null
  /** Verificações de qualidade do vencedor; reprovada = não promove. null em ciclos anteriores a elas. */
  qualidade: { aprovada: boolean; falhas: string[] } | null
  decisao: Decisao
  motivo: string
  /** Vigente ao fim do ciclo. */
  vigente_apos: RefModelo | null
}

export interface RefVencedor {
  modelo: Papel
  algoritmo: Algoritmo
  run_id: RunId
  f1_val: number
}

export interface CiclosJson extends Envelope {
  /** Ordem cronológica (mais antigo primeiro). */
  ciclos: Ciclo[]
}

// --------------------------------------------------------------------------
// votacao.json — tela Votação
// --------------------------------------------------------------------------

export interface MetodoSelecao {
  /** Chave usada em `votos` e `ranks`, ex.: "random_forest". */
  id: string
  /** Rótulo para a coluna, ex.: "Random Forest". */
  nome: string
}

export interface LinhaVoto {
  feature: string
  /** { [metodo.id]: 0 | 1 } */
  votos: Record<string, 0 | 1>
  /** { [metodo.id]: rank } — 1 = melhor. */
  ranks: Record<string, number>
  total_votos: number
  rank_medio: number
  eleita: boolean
  motivo: 'votos_suficientes' | 'completou_minimo' | null
}

export interface VotacaoCiclo {
  cycle_id: CycleId
  data_version: DataVersion
  mlflow_run_id: RunId | null
  n_treino: number
  resumo: {
    n_features_avaliadas: number
    n_eleitas: number
    n_eleitas_por_votos: number
    n_completadas_pelo_minimo: number
    n_com_3_votos: number
    n_com_2_votos: number
    n_com_1_voto: number
    n_sem_votos: number
  }
  eleitas: string[]
  /** Todas as features avaliadas; eleitas primeiro. */
  tabela: LinhaVoto[]
}

export interface VotacaoJson extends Envelope {
  regra: {
    top_k: number
    votos_minimos: number
    desempate: string
    minimo_features: number
  }
  metodos: MetodoSelecao[]
  ciclos: VotacaoCiclo[]
}

// --------------------------------------------------------------------------
// competicao.json — tela Competição
// --------------------------------------------------------------------------

export interface ModeloCompeticao {
  papel: Papel | 'vigente'
  algoritmo: Algoritmo
  run_id: RunId
  /** Ciclo em que foi treinado (A e B = o próprio ciclo). */
  cycle_id_treino: CycleId
  features: string[]
  metricas: Metricas
  matriz_confusao: MatrizConfusao
}

export interface CompeticaoCiclo {
  cycle_id: CycleId
  data_version: DataVersion
  n_validacao: number
  vencedor: Papel
  decisao: Decisao
  /** A, B e, a partir do 2º ciclo, "vigente". */
  modelos: ModeloCompeticao[]
}

export interface CompeticaoJson extends Envelope {
  /** Métricas das barras agrupadas, nesta ordem. */
  metricas: Metrica[]
  ciclos: CompeticaoCiclo[]
}

// --------------------------------------------------------------------------
// vigente.json — tela Visão geral
// --------------------------------------------------------------------------

export interface ModeloVigente extends RefModelo {
  data_version: DataVersion
  features: string[]
  f1_val: number
  f1_teste: number | null
  promovido_em: DataHora
}

export interface UltimoCiclo {
  cycle_id: CycleId
  executado_em: DataHora
  data_version: DataVersion
  decisao: Decisao
  motivo: string
  vencedor: RefVencedor
  f1_vigente_anterior: number | null
}

export interface VigenteJson extends Envelope {
  regra_promocao: { metrica: 'f1'; margem_minima: number; f1_minimo: number }
  vigente: ModeloVigente | null
  ultimo_ciclo: UltimoCiclo | null
  /** true → mostrar o banner "modelo mudou no último ciclo". */
  modelo_mudou: boolean
  /** Vigente substituído no último ciclo (null se não houve troca ou era o 1º modelo). */
  anterior: (RefModelo & { f1_val: number }) | null
}

// --------------------------------------------------------------------------
// previsoes.json — tela Previsões
// --------------------------------------------------------------------------

export interface LinhaPrevisao {
  address: string
  lote: Lote
  probabilidade: number
  previsao: 0 | 1
  rotulo_real: 0 | 1
}

export interface PrevisoesCiclo {
  cycle_id: CycleId
  data_version: DataVersion
  modelo: RefModelo
  particao: 'teste'
  limiar: number
  /** Tamanho do teste; métricas e matriz usam todas as linhas, não só a amostra. */
  n_total: number
  metricas: Metricas
  matriz_confusao: MatrizConfusao
  criterio_amostra: string
  amostra: LinhaPrevisao[]
}

export interface PrevisoesJson extends Envelope {
  ciclos: PrevisoesCiclo[]
}

// --------------------------------------------------------------------------
// Mapa arquivo → tipo (para um carregador genérico)
// --------------------------------------------------------------------------

export interface ArquivosPainel {
  ciclos: CiclosJson
  votacao: VotacaoJson
  competicao: CompeticaoJson
  vigente: VigenteJson
  previsoes: PrevisoesJson
}

export type NomeArquivo = keyof ArquivosPainel

export const ARQUIVOS_PAINEL: readonly NomeArquivo[] = [
  'ciclos',
  'votacao',
  'competicao',
  'vigente',
  'previsoes',
]

export const ROTULO_ALGORITMO: Record<Algoritmo, string> = {
  logistic_regression: 'Regressão Logística',
  xgboost: 'XGBoost',
}
