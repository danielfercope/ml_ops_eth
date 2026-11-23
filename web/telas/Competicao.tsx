import { usePainel } from '../dados/contexto'
import { EstadoVazio, Matriz, Ponto, Selo, SeletorCiclo, Tabela, Titulo } from '../componentes'
import { useCicloEscolhido } from '../ganchos'
import { GraficoBarras } from '../graficos'
import { ALG, curto, f4, int, METRICA, PAPEL_NOME } from '../formato'

export default function Competicao() {
  const { dados, rotCiclo } = usePainel()
  const comp = dados.competicao
  const ids = comp.ciclos.map((c) => c.cycle_id)
  const [atual, escolher] = useCicloEscolhido(ids)

  if (!comp.ciclos.length) {
    return (
      <>
        <Titulo titulo="Competição" />
        <EstadoVazio titulo="Nenhuma competição ainda"
          texto="A cada ciclo, A (Regressão Logística) e B (XGBoost) são treinados com as mesmas features eleitas e avaliados na mesma validação." />
      </>
    )
  }

  const c = comp.ciclos[ids.indexOf(atual)]
  const ciclo = dados.ciclos.ciclos.find((x) => x.cycle_id === c.cycle_id)
  const vig = c.modelos.find((m) => m.papel === 'vigente')
  const eleitas = c.modelos.find((m) => m.papel === 'A')?.features ?? []
  const faltam = vig ? eleitas.filter((f) => !vig.features.includes(f)) : []
  const extras = vig ? vig.features.filter((f) => !eleitas.includes(f)) : []

  return (
    <>
      <Titulo titulo="Competição"
        sub={`Validação do ${rotCiclo(c.cycle_id)}: ${int(c.n_validacao)} carteiras. Vencedor = maior F1 entre A e B.`}
        extra={<SeletorCiclo ids={ids} atual={atual} rot={rotCiclo} onEscolher={escolher} />} />
      <div className="cartao" style={{ marginBottom: 'var(--sp-4)' }}>
        <div className="barra-filtros" style={{ justifyContent: 'space-between', marginBottom: 'var(--sp-2)' }}>
          <h2 style={{ margin: 0 }}>Métricas da classe fraude na validação</h2>
          <span>
            <Selo decisao={c.decisao} />{' '}
            <span className="sec" style={{ fontSize: 'var(--fs-sm)' }}>vencedor: Modelo {c.vencedor}</span>
          </span>
        </div>
        <GraficoBarras c={c} metricas={comp.metricas} />
        {ciclo && <p className="motivo" style={{ maxWidth: 'none' }}>{ciclo.motivo}</p>}
        <Tabela cabecalho={
          <tr>
            <th>Modelo</th>
            {comp.metricas.map((m) => <th key={m} className="n">{METRICA[m]}</th>)}
            <th>Treinado no</th><th className="n">Features</th>
          </tr>
        }>
          {c.modelos.map((m) => (
            <tr key={m.papel}>
              <td>
                <Ponto papel={m.papel} />{PAPEL_NOME[m.papel]}
                <div className="muted" style={{ fontSize: 'var(--fs-xs)' }}>
                  {ALG[m.algoritmo]} · run <span className="mono">{curto(m.run_id)}</span>
                </div>
              </td>
              {comp.metricas.map((k) => (
                <td key={k} className="n">
                  {m.papel === c.vencedor && k === 'f1' ? <b>{f4(m.metricas[k])}</b> : f4(m.metricas[k])}
                </td>
              ))}
              <td>{rotCiclo(m.cycle_id_treino)}</td>
              <td className="n">{m.features.length}</td>
            </tr>
          ))}
        </Tabela>
      </div>
      <div className="grade grade--3">
        {c.modelos.map((m) => (
          <div key={m.papel} className="cartao">
            <h2>
              <Ponto papel={m.papel} />{PAPEL_NOME[m.papel]}{' '}
              <small>{ALG[m.algoritmo]}{m.papel === c.vencedor ? ' · vencedor' : ''}</small>
            </h2>
            <Matriz m={m.matriz_confusao} />
            <p className="nota">
              {m.papel === 'vigente'
                ? `Treinado no ${rotCiclo(m.cycle_id_treino)} com ${m.features.length} features${
                    faltam.length || extras.length ? ` — sem ${faltam.join(', ')}` : ', as mesmas eleitas'}.`
                : `${m.features.length} features eleitas do ${rotCiclo(c.cycle_id)}.`}
            </p>
          </div>
        ))}
      </div>
    </>
  )
}
