import { createContext, useContext } from 'react'
import type { ArquivosPainel } from './tipos'

export interface Painel {
  dados: ArquivosPainel
  /** "Ciclo N" a partir do cycle_id. */
  rotCiclo: (id: string) => string
}

export const PainelContexto = createContext<Painel | null>(null)

export function usePainel(): Painel {
  const p = useContext(PainelContexto)
  if (!p) throw new Error('usePainel fora do PainelContexto')
  return p
}
