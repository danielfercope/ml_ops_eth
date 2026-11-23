// Formatação e rótulos compartilhados pelas telas (pt-BR).
import type { Algoritmo, Ciclo, Metrica, Papel } from './dados/tipos'

export const ALG: Record<Algoritmo, string> = {
  logistic_regression: 'Regressão Logística',
  xgboost: 'XGBoost',
}
export const METRICA: Record<Metrica, string> = { precision: 'Precision', recall: 'Recall', f1: 'F1' }
export const PAPEL_NOME: Record<Papel | 'vigente', string> = {
  A: 'Modelo A',
  B: 'Modelo B',
  vigente: 'Vigente reavaliado',
}

export const f4 = (n: number | null | undefined) =>
  n == null ? '—' : n.toLocaleString('pt-BR', { minimumFractionDigits: 4, maximumFractionDigits: 4 })
export const f2 = (n: number) => n.toLocaleString('pt-BR', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
export const pct = (n: number, d = 1) =>
  (n * 100).toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d }) + '%'
export const int = (n: number) => n.toLocaleString('pt-BR')
export const fmtData = (iso: string) =>
  new Intl.DateTimeFormat('pt-BR', { dateStyle: 'short', timeStyle: 'short' }).format(new Date(iso))
export const fmtDataLonga = (iso: string) =>
  new Intl.DateTimeFormat('pt-BR', {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', timeZoneName: 'short',
  }).format(new Date(iso))
export const curto = (run: string | null | undefined) => (run ? run.slice(0, 8) : '—')

export const nomeModelo = (r: { algoritmo: Algoritmo; modelo?: Papel; papel?: Papel | 'vigente' }) =>
  `${ALG[r.algoritmo]} (${r.modelo ?? r.papel})`

/** "Ciclo N" a partir da posição do cycle_id na lista cronológica. */
export function rotuladorCiclos(ciclos: Ciclo[]) {
  return (id: string) => `Ciclo ${ciclos.findIndex((c) => c.cycle_id === id) + 1}`
}

export type TipoResultado = 'VP' | 'VN' | 'FP' | 'FN'
export const tipoPrevisao = (l: { previsao: 0 | 1; rotulo_real: 0 | 1 }): TipoResultado =>
  l.previsao ? (l.rotulo_real ? 'VP' : 'FP') : l.rotulo_real ? 'FN' : 'VN'
