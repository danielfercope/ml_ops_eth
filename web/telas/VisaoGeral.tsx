import { Link } from 'react-router'
import { usePainel } from '../dados/contexto'
import type { VigenteJson } from '../dados/tipos'
import { Chips, EstadoVazio, Icone, Kpi, Lotes, Ponto, Selo, Titulo } from '../componentes'
import { GraficoDecisao } from '../graficos'
import { ALG, curto, f2, f4, fmtData, fmtDataLonga, nomeModelo } from '../formato'

function TextoRegra({ r }: { r: VigenteJson['regra_promocao'] }) {
  return (
    <p style={{ margin: 0, fontSize: 'var(--fs-sm)' }}>
      O vencedor do ciclo substitui o vigente só se{' '}
      <span className="mono">F1_val ≥ F1_vigente + {f2(r.margem_minima)}</span> e{' '}
      <span className="mono">F1_val ≥ {f2(r.f1_minimo)}</span>. O F1 do vigente é reavaliado na validação do ciclo atual.
    </p>
  )
}

export default function VisaoGeral() {
  const { dados, rotCiclo } = usePainel()
  const vg = dados.vigente
  const regra = vg.regra_promocao

  if (!vg.vigente || !vg.ultimo_ciclo) {
    return (
      <>
        <Titulo titulo="Visão geral" />
        <EstadoVazio
          titulo="Nenhum ciclo executado ainda"
          texto={`O primeiro ciclo agendado vai treinar os modelos A (Regressão Logística) e B (XGBoost) e promover o vencedor se o F1 de validação for ≥ ${f2(regra.f1_minimo)}.`}
        />
        <div className="cartao" style={{ marginTop: 'var(--sp-4)' }}>
          <h2>Regra de promoção</h2>
          <TextoRegra r={regra} />
        </div>
      </>
    )
  }

  const v = vg.vigente, u = vg.ultimo_ciclo, ant = vg.anterior
  const ciclos = dados.ciclos.ciclos
  const ultimo = ciclos.find((c) => c.cycle_id === u.cycle_id)

  return (
    <>
      <Titulo titulo="Visão geral" sub="Modelo em produção e o que aconteceu no último ciclo." />
      {vg.modelo_mudou && (
        <div className="aviso aviso--troca" role="status">
          <Icone nome="troca" />
          <div>
            <b>{ant ? 'O modelo vigente mudou no último ciclo' : 'Primeiro modelo vigente promovido'}</b>
            {ant && `${nomeModelo(ant)} · ${rotCiclo(ant.cycle_id)} (F1 ${f4(ant.f1_val)}) → `}
            <b style={{ display: 'inline' }}>{nomeModelo(v)} · {rotCiclo(v.cycle_id)}</b> (F1 {f4(v.f1_val)})
          </div>
        </div>
      )}
      <div className="grade grade--kpi">
        <Kpi rotulo="Modelo vigente" medio valor={<><Ponto papel={v.modelo} />{ALG[v.algoritmo]}</>}
          sub={<>Modelo {v.modelo} · run <span className="mono">{curto(v.run_id)}</span></>} />
        <Kpi rotulo="F1 validação" valor={f4(v.f1_val)} sub="no ciclo em que foi treinado" />
        <Kpi rotulo="F1 teste" valor={f4(v.f1_teste)} sub="notebook 05 · teste inteiro" />
        <Kpi rotulo="Treinado no" medio valor={rotCiclo(v.cycle_id)} sub={v.cycle_id} subMono />
        <Kpi rotulo="data_version" medio valor={<span className="mono">{v.data_version}</span>}
          sub={`promovido em ${fmtData(v.promovido_em)}`} />
      </div>
      <div className="grade grade--2">
        <div className="cartao">
          <h2>Última decisão <small>{rotCiclo(u.cycle_id)} · {fmtDataLonga(u.executado_em)}</small></h2>
          <p style={{ margin: '0 0 var(--sp-2)' }}>
            <Selo decisao={u.decisao} />{' '}
            <span className="sec" style={{ marginLeft: 6 }}>
              vencedor: <Ponto papel={u.vencedor.modelo} />{nomeModelo(u.vencedor)}
            </span>
          </p>
          <p className="sec" style={{ margin: '0 0 var(--sp-3)', fontSize: 'var(--fs-sm)' }}>{u.motivo}</p>
          <GraficoDecisao venc={u.vencedor} f1Vig={u.f1_vigente_anterior} regra={regra} />
          {ultimo && (
            <dl className="dl" style={{ marginTop: 'var(--sp-3)' }}>
              <dt>Lotes</dt>
              <dd>
                <Lotes lotes={ultimo.lotes} novos={ultimo.lotes_novos} />
                {!ultimo.lotes_novos.length && <span className="muted"> sem dados novos</span>}
              </dd>
              <dt>Execução</dt>
              <dd>
                {ultimo.execucao_url ? <a href={ultimo.execucao_url} target="_blank" rel="noopener">GitHub Actions</a> : '—'}{' '}
                <span className="muted">({ultimo.gatilho ?? 'gatilho desconhecido'})</span>
              </dd>
            </dl>
          )}
        </div>
        <div className="pilha">
          <div className="cartao">
            <h2>Features do vigente <small>{v.features.length}</small></h2>
            <Chips itens={v.features} />
          </div>
          <div className="cartao">
            <h2>Histórico <small><Link to="/ciclos">ver ciclos</Link></small></h2>
            <ul className="lista-ciclos">
              {ciclos.slice().reverse().map((c) => (
                <li key={c.cycle_id}>
                  <span><b>{rotCiclo(c.cycle_id)}</b> <span className="muted">{fmtData(c.executado_em)}</span></span>
                  <span><Selo decisao={c.decisao} /></span>
                </li>
              ))}
            </ul>
          </div>
          <div className="cartao">
            <h2>Regra de promoção</h2>
            <TextoRegra r={regra} />
          </div>
        </div>
      </div>
    </>
  )
}
