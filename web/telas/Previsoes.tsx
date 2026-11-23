import { useState } from 'react'
import { usePainel } from '../dados/contexto'
import { EstadoVazio, Kpi, Matriz, Ponto, SeletorCiclo, SeloResultado, Tabela, Titulo } from '../componentes'
import { useCicloEscolhido } from '../ganchos'
import { ALG, f2, f4, int, METRICA, tipoPrevisao, type TipoResultado } from '../formato'

type Filtro = 'todos' | TipoResultado
const FILTROS: Filtro[] = ['todos', 'VP', 'VN', 'FP', 'FN']

export default function Previsoes() {
  const { dados, rotCiclo } = usePainel()
  const prev = dados.previsoes
  const ids = prev.ciclos.map((c) => c.cycle_id)
  const [atual, escolher] = useCicloEscolhido(ids)
  const [filtro, setFiltro] = useState<Filtro>('todos')

  if (!prev.ciclos.length) {
    return (
      <>
        <Titulo titulo="Previsões" />
        <EstadoVazio titulo="Sem previsões ainda"
          texto="Depois que um modelo for promovido, o notebook 05 prevê o conjunto de teste com ele e uma amostra aparece aqui." />
      </>
    )
  }

  const c = prev.ciclos[ids.indexOf(atual)]
  const contagem: Record<Filtro, number> = { todos: c.amostra.length, VP: 0, VN: 0, FP: 0, FN: 0 }
  c.amostra.forEach((l) => contagem[tipoPrevisao(l)]++)
  const linhas = c.amostra.filter((l) => filtro === 'todos' || tipoPrevisao(l) === filtro)
  const m = c.modelo

  return (
    <>
      <Titulo titulo="Previsões"
        sub="Notebook 05: o modelo vigente do ciclo prevê o conjunto de teste, que nunca é usado nas escolhas."
        extra={<SeletorCiclo ids={ids} atual={atual} rot={rotCiclo} onEscolher={escolher} />} />
      <div className="grade grade--kpi">
        <Kpi rotulo="Modelo usado" medio valor={<><Ponto papel={m.modelo} />{ALG[m.algoritmo]}</>}
          sub={`Modelo ${m.modelo} · treinado no ${rotCiclo(m.cycle_id)}`} />
        <Kpi rotulo="Carteiras no teste" valor={int(c.n_total)} sub={c.data_version} subMono />
        {(['precision', 'recall', 'f1'] as const).map((k) => (
          <Kpi key={k} rotulo={`${METRICA[k]} teste`} valor={f4(c.metricas[k])} sub={`limiar ${f2(c.limiar)}`} />
        ))}
      </div>
      <div className="grade grade--2">
        <div className="cartao">
          <div className="barra-filtros" style={{ justifyContent: 'space-between' }}>
            <h2 style={{ margin: 0 }}>Amostra <small>{c.amostra.length} de {int(c.n_total)}</small></h2>
            <div className="segmentado" role="group" aria-label="Filtrar por resultado">
              {FILTROS.map((k) => (
                <button key={k} type="button" aria-pressed={filtro === k} onClick={() => setFiltro(k)}>
                  {k === 'todos' ? 'Todos' : k} <span className="muted">{contagem[k]}</span>
                </button>
              ))}
            </div>
          </div>
          <Tabela cabecalho={
            <tr><th>Address</th><th>Lote</th><th>Probabilidade de fraude</th><th>Previsão</th><th>Real</th><th>Resultado</th></tr>
          }>
            {linhas.length ? linhas.map((l) => (
              <tr key={l.address}>
                <td className="mono" title={l.address}>{l.address.slice(0, 10)}…{l.address.slice(-6)}</td>
                <td><span className="etiqueta">{l.lote}</span></td>
                <td>
                  <div className="prob">
                    <div className="prob__trilho">
                      <div className="prob__fill" style={{ width: `${l.probabilidade * 100}%` }} />
                      <div className="prob__limiar" style={{ left: `${c.limiar * 100}%` }} title={`limiar ${f2(c.limiar)}`} />
                    </div>
                    <span className="num" style={{ width: 52, textAlign: 'right' }}>{f4(l.probabilidade)}</span>
                  </div>
                </td>
                <td>{l.previsao ? <b>Fraude</b> : 'Legítima'}</td>
                <td>{l.rotulo_real ? <b>Fraude</b> : 'Legítima'}</td>
                <td><SeloResultado tipo={tipoPrevisao(l)} /></td>
              </tr>
            )) : (
              <tr><td colSpan={6} className="muted">Nenhuma linha com esse resultado na amostra.</td></tr>
            )}
          </Tabela>
          <p className="nota">Amostra: {c.criterio_amostra}. A barra vertical na probabilidade marca o limiar.</p>
        </div>
        <div className="pilha">
          <div className="cartao">
            <h2>Matriz de confusão <small>teste inteiro</small></h2>
            <Matriz m={c.matriz_confusao} />
          </div>
          <div className="cartao">
            <h2>Modelo</h2>
            <dl className="dl">
              <dt>Algoritmo</dt><dd>{ALG[m.algoritmo]} (Modelo {m.modelo})</dd>
              <dt>Treinado no</dt><dd>{rotCiclo(m.cycle_id)} <span className="mono muted">{m.cycle_id}</span></dd>
              <dt>run_id</dt><dd className="mono">{m.run_id}</dd>
            </dl>
            <p className="nota">O teste cresce quando entra um lote novo; métricas de ciclos diferentes não são sobre o mesmo conjunto.</p>
          </div>
        </div>
      </div>
    </>
  )
}
