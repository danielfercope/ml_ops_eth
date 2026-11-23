import { useState } from 'react'
import { usePainel } from '../dados/contexto'
import type { LinhaVoto } from '../dados/tipos'
import { EstadoVazio, SeletorCiclo, Tabela, Titulo } from '../componentes'
import { useCicloEscolhido } from '../ganchos'
import { curto, f2, int } from '../formato'

function ChipsInline({ itens }: { itens: string[] }) {
  return <>{itens.map((f) => <span key={f}><span className="chip">{f}</span> </span>)}</>
}

export default function Votacao() {
  const { dados, rotCiclo } = usePainel()
  const vot = dados.votacao
  const ids = vot.ciclos.map((c) => c.cycle_id)
  const [atual, escolher] = useCicloEscolhido(ids)
  const [ocultarSemVoto, setOcultar] = useState(false)

  if (!vot.ciclos.length) {
    return (
      <>
        <Titulo titulo="Votação de features" />
        <EstadoVazio titulo="Nenhuma votação ainda"
          texto={`A cada ciclo, ${vot.metodos.map((m) => m.nome).join(', ')} votam nas top-${vot.regra.top_k} features; entra quem tiver ≥ ${vot.regra.votos_minimos} votos.`} />
      </>
    )
  }

  const idx = ids.indexOf(atual)
  const c = vot.ciclos[idx]
  const anterior = idx > 0 ? vot.ciclos[idx - 1] : null
  const antMap: Record<string, LinhaVoto> | null = anterior
    ? Object.fromEntries(anterior.tabela.map((l) => [l.feature, l]))
    : null
  const entraram = anterior ? c.eleitas.filter((f) => !anterior.eleitas.includes(f)) : []
  const sairam = anterior ? anterior.eleitas.filter((f) => !c.eleitas.includes(f)) : []
  const r = c.resumo, regra = vot.regra

  const delta = (l: LinhaVoto) => {
    const a = antMap?.[l.feature]
    if (a && a.eleita !== l.eleita) return <span className="delta delta--entrou">{l.eleita ? '▲ entrou' : '▼ saiu'}</span>
    if (a && a.total_votos !== l.total_votos)
      return <span className="delta">{l.total_votos > a.total_votos ? '+' : '−'}{Math.abs(l.total_votos - a.total_votos)} voto</span>
    return <span className="muted">—</span>
  }

  return (
    <>
      <Titulo titulo="Votação de features"
        sub={`Três métodos votam nas top-${regra.top_k}; eleita com ≥ ${regra.votos_minimos} de ${vot.metodos.length} votos, desempate por rank médio, mínimo de ${regra.minimo_features}.`}
        extra={<SeletorCiclo ids={ids} atual={atual} rot={rotCiclo} onEscolher={escolher} />} />
      <div className="cartao" style={{ marginBottom: 'var(--sp-4)' }}>
        <h2>
          {rotCiclo(c.cycle_id)}{' '}
          <small>
            <span className="mono">{c.data_version}</span> · treino {int(c.n_treino)} carteiras
            {c.mlflow_run_id && <> · run <span className="mono">{curto(c.mlflow_run_id)}</span></>}
          </small>
        </h2>
        <div className="resumo-votos">
          <div><b>{r.n_eleitas}</b><span>eleitas de {r.n_features_avaliadas}</span></div>
          <div><b>{r.n_com_3_votos}</b><span>com 3 votos</span></div>
          <div><b>{r.n_com_2_votos}</b><span>com 2 votos</span></div>
          <div><b>{r.n_com_1_voto}</b><span>com 1 voto</span></div>
          <div><b>{r.n_sem_votos}</b><span>sem voto</span></div>
          <div><b>{r.n_completadas_pelo_minimo}</b><span>completadas pelo mínimo</span></div>
        </div>
        <div className="chips">{c.eleitas.map((f) => <span key={f} className="chip">{f}</span>)}</div>
        {anterior && (
          <p style={{ margin: 'var(--sp-3) 0 0', fontSize: 'var(--fs-sm)' }}>
            <b>Em relação ao {rotCiclo(anterior.cycle_id)}:</b>{' '}
            {entraram.length || sairam.length ? (
              <>
                {entraram.length > 0 && <>entrou <ChipsInline itens={entraram} /></>}
                {sairam.length > 0 && <>saiu <ChipsInline itens={sairam} /></>}
              </>
            ) : (
              <span className="sec">mesmo conjunto de features eleitas.</span>
            )}
          </p>
        )}
      </div>
      <div className="cartao">
        <div className="barra-filtros" style={{ justifyContent: 'space-between' }}>
          <h2 style={{ margin: 0 }}>Tabela de votos <small>célula = rank no método; ✓ = voto (top-{regra.top_k})</small></h2>
          <label className="controle">
            <input type="checkbox" checked={ocultarSemVoto} onChange={(e) => setOcultar(e.target.checked)} /> ocultar
            features sem voto
          </label>
        </div>
        <Tabela cabecalho={
          <tr>
            <th>Feature</th>
            {vot.metodos.map((m) => <th key={m.id} className="c">{m.nome}</th>)}
            <th className="c">Votos</th><th className="n">Rank médio</th><th>Eleita</th>
            {anterior && <th>vs {rotCiclo(anterior.cycle_id)}</th>}
          </tr>
        }>
          {c.tabela.filter((l) => !ocultarSemVoto || l.total_votos > 0).map((l) => (
            <tr key={l.feature} className={l.eleita ? 'eleita' : ''}>
              <td className="mono" style={{ overflowWrap: 'anywhere', minWidth: 220 }}>{l.feature}</td>
              {vot.metodos.map((m) => (
                <td key={m.id} className="c">
                  <span className={`voto voto--${l.votos[m.id] ? 'sim' : 'nao'}`}
                    title={`${m.nome}: rank ${l.ranks[m.id]}${l.votos[m.id] ? ` (no top-${regra.top_k})` : ''}`}>
                    {l.votos[m.id] ? '✓ ' : ''}{l.ranks[m.id]}
                  </span>
                </td>
              ))}
              <td className="c">
                <span className="pips" aria-label={`${l.total_votos} votos`}>
                  {vot.metodos.map((m, i) => <i key={m.id} className={i < l.total_votos ? 'on' : ''} />)}
                </span>{' '}
                <span className="num">{l.total_votos}</span>
              </td>
              <td className="n">{f2(l.rank_medio)}</td>
              <td>
                {l.eleita && (l.motivo === 'completou_minimo' ? <span className="etiqueta">completou mínimo</span> : <b>Eleita</b>)}
              </td>
              {anterior && <td>{delta(l)}</td>}
            </tr>
          ))}
        </Tabela>
      </div>
    </>
  )
}
