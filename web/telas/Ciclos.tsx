import { usePainel } from '../dados/contexto'
import { EstadoVazio, Lotes, Ponto, Selo, Tabela, Titulo } from '../componentes'
import { GraficoCiclos } from '../graficos'
import { curto, f4, fmtData, int, nomeModelo, pct } from '../formato'

const PEQ = { fontSize: 'var(--fs-xs)' }

export default function Ciclos() {
  const { dados, rotCiclo } = usePainel()
  const ciclos = dados.ciclos.ciclos
  if (!ciclos.length) {
    return (
      <>
        <Titulo titulo="Ciclos" />
        <EstadoVazio titulo="Nenhum ciclo ainda"
          texto="Cada execução do workflow agendado aparece aqui com os dados usados, o vencedor e a decisão." />
      </>
    )
  }
  return (
    <>
      <Titulo titulo="Ciclos"
        sub={`${ciclos.length} ${ciclos.length === 1 ? 'execução' : 'execuções'} do pipeline, do mais recente para o mais antigo.`} />
      <div className="cartao" style={{ marginBottom: 'var(--sp-4)' }}>
        <h2>F1 de validação por ciclo <small>vencedor × vigente reavaliado × exigido</small></h2>
        <GraficoCiclos ciclos={ciclos} regra={dados.vigente.regra_promocao} rot={rotCiclo} />
        <p className="nota">
          A validação cresce quando entra um lote novo, por isso o vigente é reavaliado a cada ciclo: só os pontos de um
          mesmo ciclo são comparáveis entre si.
        </p>
      </div>
      <div className="cartao">
        <Tabela cabecalho={
          <tr>
            <th>Ciclo</th><th>Executado</th><th>Dados</th>
            <th className="n">Carteiras<br /><span style={{ textTransform: 'none' }}>treino / val / teste</span></th>
            <th className="n">Fraude<br />treino</th><th className="n">Features</th><th>Vencedor</th>
            <th className="n">F1 val</th><th className="n">F1 vigente<br />reavaliado</th><th>Decisão</th>
          </tr>
        }>
          {ciclos.slice().reverse().map((c) => (
            <tr key={c.cycle_id}>
              <td><b>{rotCiclo(c.cycle_id)}</b><div className="mono muted" style={PEQ}>{c.cycle_id}</div></td>
              <td className="num">
                {fmtData(c.executado_em)}
                <div className="muted" style={PEQ}>
                  {c.execucao_url
                    ? <a href={c.execucao_url} target="_blank" rel="noopener">{c.gatilho ?? 'execução'}</a>
                    : c.gatilho ?? '—'}
                </div>
              </td>
              <td>
                <span className="mono">{c.data_version}</span>
                <div style={{ marginTop: 4 }}><Lotes lotes={c.lotes} novos={c.lotes_novos} /></div>
                {!c.lotes_novos.length && <div className="muted" style={PEQ}>sem dados novos</div>}
              </td>
              <td className="n">
                {int(c.dados.n_carteiras)}
                <div className="muted" style={PEQ}>
                  {int(c.dados.n_treino)} / {int(c.dados.n_validacao)} / {int(c.dados.n_teste)}
                </div>
              </td>
              <td className="n">{pct(c.dados.taxa_fraude_treino)}</td>
              <td className="n">{c.n_features_eleitas}</td>
              <td>
                <Ponto papel={c.vencedor.modelo} />{nomeModelo(c.vencedor)}
                <div className="mono muted" style={PEQ}>run {curto(c.vencedor.run_id)}</div>
              </td>
              <td className="n"><b>{f4(c.vencedor.f1_val)}</b></td>
              <td className="n">{c.vigente_anterior ? f4(c.vigente_anterior.f1_val) : <span className="muted">—</span>}</td>
              <td>
                <Selo decisao={c.decisao} />
                {c.qualidade && !c.qualidade.aprovada && <> <span className="etiqueta">qualidade reprovada</span></>}
                <div className="motivo">{c.motivo}</div>
              </td>
            </tr>
          ))}
        </Tabela>
        <p className="nota">
          Carteiras = total após remover endereços duplicados (acumulado:{' '}
          {ciclos.map((c) => `${rotCiclo(c.cycle_id)} ${c.dados.duplicados_removidos}`).join(', ')}).
        </p>
      </div>
    </>
  )
}
