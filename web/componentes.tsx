// Componentes compartilhados pelas telas: cartão de KPI, selo de status,
// tabela, matriz de confusão, seletor de ciclo e estados de carregamento/erro/vazio.
import { useEffect, useRef, type ReactNode } from 'react'
import type { Decisao, Lote, MatrizConfusao, Papel } from './dados/tipos'
import { int, type TipoResultado } from './formato'

// --------------------------------------------------------------------------
// Ícones
// --------------------------------------------------------------------------
const ICONES = {
  check: (
    <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 8.5l3 3 7-7" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  igual: (
    <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M3 6h10M3 10h10" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  ),
  x: (
    <svg width="12" height="12" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M4 4l8 8M12 4l-8 8" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  ),
  info: (
    <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
      <circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M8 7v4.5M8 4.6v.1" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  ),
  troca: (
    <svg width="18" height="18" viewBox="0 0 16 16" aria-hidden="true">
      <path d="M2 5h10l-3-3M14 11H4l3 3" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  vazio: (
    <svg width="40" height="40" viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="4" width="18" height="16" rx="2" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M3 9h18M8 14h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  ),
  erro: (
    <svg width="40" height="40" viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="1.5" />
      <path d="M12 7.5v5.5M12 16.2v.1" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  ),
}
export type NomeIcone = keyof typeof ICONES
export const Icone = ({ nome }: { nome: NomeIcone }) => ICONES[nome]

// --------------------------------------------------------------------------
// Selos, pontos e etiquetas
// --------------------------------------------------------------------------
export function Selo({ decisao }: { decisao: Decisao }) {
  return decisao === 'promovido' ? (
    <span className="selo selo--promovido"><Icone nome="check" />Promovido</span>
  ) : (
    <span className="selo selo--mantido"><Icone nome="igual" />Mantido</span>
  )
}

export function SeloResultado({ tipo }: { tipo: TipoResultado }) {
  const ok = tipo === 'VP' || tipo === 'VN'
  return (
    <span className={`selo selo--${ok ? 'acerto' : 'erro'}`}>
      <Icone nome={ok ? 'check' : 'x'} />{tipo}
    </span>
  )
}

export const Ponto = ({ papel }: { papel: Papel | 'vigente' }) => <span className={`ponto ponto--${papel}`} />

export function Lotes({ lotes, novos }: { lotes: Lote[]; novos: Lote[] }) {
  return (
    <>
      {lotes.map((l, i) => (
        <span key={l}>
          {i > 0 && ' '}
          <span className={`etiqueta ${novos.includes(l) ? 'etiqueta--novo' : ''}`}>
            {l}{novos.includes(l) ? ' · novo' : ''}
          </span>
        </span>
      ))}
    </>
  )
}

export function Chips({ itens }: { itens: string[] }) {
  return <div className="chips">{itens.map((f) => <span key={f} className="chip">{f}</span>)}</div>
}

// --------------------------------------------------------------------------
// Estrutura
// --------------------------------------------------------------------------
export function Titulo({ titulo, sub, extra }: { titulo: string; sub?: ReactNode; extra?: ReactNode }) {
  return (
    <div className="titulo">
      <div>
        <h1>{titulo}</h1>
        {sub && <p>{sub}</p>}
      </div>
      {extra}
    </div>
  )
}

export function Kpi({ rotulo, valor, sub, medio, subMono }: {
  rotulo: string; valor: ReactNode; sub?: ReactNode; medio?: boolean; subMono?: boolean
}) {
  return (
    <div className="kpi">
      <div className="kpi__rotulo">{rotulo}</div>
      <div className={`kpi__valor ${medio ? 'kpi__valor--md' : ''}`}>{valor}</div>
      {sub != null && <div className={`kpi__sub ${subMono ? 'mono' : ''}`}>{sub}</div>}
    </div>
  )
}

export function Tabela({ cabecalho, children }: { cabecalho: ReactNode; children: ReactNode }) {
  return (
    <div className="tabela-wrap">
      <table>
        <thead>{cabecalho}</thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  )
}

export function Matriz({ m }: { m: MatrizConfusao }) {
  return (
    <div className="matriz" role="table" aria-label="Matriz de confusão">
      <div />
      <div className="cab">prev. legítima</div>
      <div className="cab">prev. fraude</div>
      <div className="cab cab--lin">real legítima</div>
      <div className="cel cel--ok"><b>{int(m.vn)}</b><small>VN</small></div>
      <div className="cel cel--erro"><b>{int(m.fp)}</b><small>FP</small></div>
      <div className="cab cab--lin">real fraude</div>
      <div className="cel cel--erro"><b>{int(m.fn)}</b><small>FN</small></div>
      <div className="cel cel--ok"><b>{int(m.vp)}</b><small>VP</small></div>
    </div>
  )
}

/** Botões "Ciclo 1 | Ciclo 2 | ..."; some quando há menos de 2 ciclos. */
export function SeletorCiclo({ ids, atual, rot, onEscolher }: {
  ids: string[]; atual: string; rot: (id: string) => string; onEscolher: (id: string) => void
}) {
  if (ids.length < 2) return null
  return (
    <div className="segmentado" role="group" aria-label="Ciclo">
      {ids.map((id) => (
        <button key={id} type="button" aria-pressed={id === atual} title={id} onClick={() => onEscolher(id)}>
          {rot(id)}
        </button>
      ))}
    </div>
  )
}

// --------------------------------------------------------------------------
// Estados
// --------------------------------------------------------------------------
export function EstadoVazio({ titulo, texto }: { titulo: string; texto: ReactNode }) {
  return (
    <div className="cartao estado">
      <Icone nome="vazio" />
      <h2>{titulo}</h2>
      <p>{texto}</p>
    </div>
  )
}

export function Carregando() {
  const bloco = (h: number, k: number) => <div key={k} className="esqueleto" style={{ height: h }} />
  return (
    <div aria-busy="true">
      <span className="sr-only">Carregando dados do painel…</span>
      <div className="esqueleto" style={{ height: 28, width: 220, marginBottom: 'var(--sp-4)' }} />
      <div className="grade grade--kpi">{[0, 1, 2, 3].map((k) => bloco(96, k))}</div>
      <div className="grade grade--2">{bloco(280, 0)}{bloco(280, 1)}</div>
    </div>
  )
}

export function ErroCarregamento({ erro, onTentar, onArquivos }: {
  erro: string; onTentar: () => void; onArquivos: () => void
}) {
  return (
    <div className="cartao estado" role="alert">
      <Icone nome="erro" />
      <h2>Não foi possível carregar os dados do painel</h2>
      <pre className="erro">{erro}</pre>
      <p>
        Sirva o repositório por HTTP (<span className="mono">npm run dev</span>) ou escolha à mão os 5 JSONs de uma
        pasta de cenário.
      </p>
      <p>
        <button className="botao" type="button" onClick={onTentar}>Tentar de novo</button>{' '}
        <button className="botao" type="button" onClick={onArquivos}>Carregar JSONs…</button>
      </p>
    </div>
  )
}

// --------------------------------------------------------------------------
// Tooltip dos gráficos
// --------------------------------------------------------------------------
/** Tooltip global: qualquer elemento com data-tip mostra o texto ao passar o mouse. */
export function Tooltip() {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const mover = (e: MouseEvent) => {
      const tip = ref.current
      if (!tip) return
      const alvo = (e.target as Element | null)?.closest?.('[data-tip]')
      if (!alvo) {
        tip.classList.remove('on')
        return
      }
      tip.textContent = alvo.getAttribute('data-tip')
      tip.style.left = `${Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8)}px`
      tip.style.top = `${e.clientY + 14}px`
      tip.classList.add('on')
    }
    document.addEventListener('mousemove', mover)
    return () => document.removeEventListener('mousemove', mover)
  }, [])
  return <div className="tooltip" ref={ref} role="tooltip" />
}
