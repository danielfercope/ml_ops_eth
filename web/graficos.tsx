// Gráficos SVG das telas (portados do protótipo). Desenhados em 1:1 com a
// largura medida do contêiner, para o texto não escalar.
import type { Ciclo, CompeticaoCiclo, Metrica, RefVencedor, VigenteJson } from './dados/tipos'
import { Ponto } from './componentes'
import { useLargura } from './ganchos'
import { ALG, f2, f4, METRICA, nomeModelo, PAPEL_NOME } from './formato'

type Regra = VigenteJson['regra_promocao']

function ticks(lo: number, hi: number, passo: number): number[] {
  const out: number[] = []
  for (let t = Math.ceil(lo / passo) * passo; t <= hi + 1e-9; t += passo) out.push(+t.toFixed(2))
  return out
}

// --------------------------------------------------------------------------
// Régua da decisão: vigente reavaliado, F1 exigido e vencedor
// --------------------------------------------------------------------------
export function GraficoDecisao({ venc, f1Vig, regra }: { venc: RefVencedor; f1Vig: number | null; regra: Regra }) {
  const [ref, largura] = useLargura<HTMLDivElement>()
  const exigido = f1Vig == null ? regra.f1_minimo : Math.max(f1Vig + regra.margem_minima, regra.f1_minimo)
  const vals = [venc.f1_val, exigido, ...(f1Vig == null ? [] : [f1Vig])]
  const lo = Math.floor((Math.min(...vals) - 0.02) * 100) / 100
  const hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.01) * 100) / 100)
  const W = Math.min(520, largura), H = 92, mx = 12, y = 40
  const x = (v: number) => mx + ((v - lo) / (hi - lo)) * (W - 2 * mx)
  const passo = hi - lo > 0.2 ? 0.1 : hi - lo > 0.06 ? 0.02 : 0.01
  const passou = venc.f1_val >= exigido - 1e-9
  const junto = f1Vig != null && Math.abs(x(f1Vig) - x(venc.f1_val)) < 14
  const yVig = junto ? y - 8 : y, yVenc = junto ? y + 8 : y
  return (
    <div ref={ref}>
      <div className="legenda" style={{ marginBottom: 'var(--sp-1)' }}>
        <span><Ponto papel={venc.modelo} />Vencedor ({venc.modelo})</span>
        {f1Vig != null && <span><Ponto papel="vigente" />Vigente reavaliado</span>}
        <span><span className="traco" />F1 exigido</span>
      </div>
      <svg className="grafico grafico--regua" viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`F1 do vencedor ${f4(venc.f1_val)} contra F1 exigido ${f4(exigido)}`}>
        <line className="eixo-forte" x1={mx} x2={W - mx} y1={y} y2={y} strokeWidth={2} />
        {ticks(lo, hi, passo).map((t) => (
          <g key={t}>
            <line className="grade-linha" x1={x(t)} x2={x(t)} y1={y - 4} y2={y + 4} />
            <text x={x(t)} y={y + 20} textAnchor="middle">{f2(t)}</text>
          </g>
        ))}
        <rect x={x(exigido)} y={y - 10} width={Math.max(0, x(hi) - x(exigido))} height={20}
          fill="var(--color-status-good)" opacity={0.12} />
        <line x1={x(exigido)} x2={x(exigido)} y1={y - 18} y2={y + 10} stroke="var(--color-text-2)" strokeWidth={2}
          strokeDasharray="4 3" data-tip={`F1 exigido: ${f4(exigido)}`} />
        <text x={x(exigido)} y={y - 22} textAnchor="middle">exigido {f4(exigido)}</text>
        {f1Vig != null && (
          <circle cx={x(f1Vig)} cy={yVig} r={7} fill="var(--color-modelo-vigente)" stroke="var(--color-surface)"
            strokeWidth={2} data-tip={`Vigente reavaliado: F1 ${f4(f1Vig)}`} />
        )}
        <circle cx={x(venc.f1_val)} cy={yVenc} r={7} fill={`var(--color-modelo-${venc.modelo.toLowerCase()})`}
          stroke="var(--color-surface)" strokeWidth={2}
          data-tip={`Vencedor ${nomeModelo(venc)}: F1 ${f4(venc.f1_val)} · ${passou ? 'atinge' : 'não atinge'} o exigido`} />
        <text className="rotulo-valor" x={x(venc.f1_val)} y={y + 40} textAnchor="middle">{f4(venc.f1_val)}</text>
      </svg>
    </div>
  )
}

// --------------------------------------------------------------------------
// F1 de validação por ciclo: vencedor × vigente reavaliado × exigido
// --------------------------------------------------------------------------
export function GraficoCiclos({ ciclos, regra, rot }: { ciclos: Ciclo[]; regra: Regra; rot: (id: string) => string }) {
  const [ref, largura] = useLargura<HTMLDivElement>()
  const pts = ciclos.map((c) => {
    const ant = c.vigente_anterior
    const exigido = ant ? Math.max(ant.f1_val + regra.margem_minima, regra.f1_minimo) : regra.f1_minimo
    return { c, venc: c.vencedor.f1_val, vig: ant ? ant.f1_val : null, exigido }
  })
  const vals = pts.flatMap((p) => [p.venc, p.vig, p.vig != null ? p.exigido : null]).filter((v): v is number => v != null)
  const lo = Math.floor((Math.min(...vals) - 0.015) * 100) / 100
  const hi = Math.min(1, Math.ceil((Math.max(...vals) + 0.01) * 100) / 100)
  const W = largura, H = 230, ml = 48, mr = 16, mt = 12, mb = 46
  const banda = (W - ml - mr) / pts.length
  const xc = (i: number) => ml + banda * (i + 0.5)
  const y = (v: number) => mt + ((hi - v) / (hi - lo)) * (H - mt - mb)
  const passo = hi - lo > 0.08 ? 0.02 : 0.01
  const temA = pts.some((p) => p.c.vencedor.modelo === 'A'), temB = pts.some((p) => p.c.vencedor.modelo === 'B')
  return (
    <div ref={ref}>
      <div className="legenda">
        {temA && <span><Ponto papel="A" />Vencedor: Modelo A</span>}
        {temB && <span><Ponto papel="B" />Vencedor: Modelo B</span>}
        <span><Ponto papel="vigente" />Vigente reavaliado</span>
        <span><span className="traco" />F1 exigido para promover</span>
      </div>
      <svg className="grafico" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="F1 de validação do vencedor e do vigente por ciclo">
        {ticks(lo, hi, passo).map((t) => (
          <g key={t}>
            <line className="grade-linha" x1={ml} x2={W - mr} y1={y(t)} y2={y(t)} />
            <text x={ml - 8} y={y(t) + 4} textAnchor="end">{f2(t)}</text>
          </g>
        ))}
        <line className="eixo-forte" x1={ml} x2={W - mr} y1={H - mb} y2={H - mb} />
        {pts.map((p, i) => {
          const sobrepoe = p.vig != null && Math.abs(y(p.vig) - y(p.venc)) < 14
          const cx = xc(i) + (sobrepoe ? 8 : 0), cxv = xc(i) - (sobrepoe ? 8 : 0)
          const cor = `var(--color-modelo-${p.c.vencedor.modelo.toLowerCase()})`
          const ant = p.c.vigente_anterior
          return (
            <g key={p.c.cycle_id}>
              {p.vig != null && ant ? (
                <>
                  <line x1={cxv} x2={cx} y1={y(p.vig)} y2={y(p.venc)} stroke="var(--color-border-strong)" strokeWidth={2} />
                  <line x1={cx - 22} x2={cx + 22} y1={y(p.exigido)} y2={y(p.exigido)} stroke="var(--color-text-2)"
                    strokeWidth={2} strokeDasharray="4 3" />
                  <rect x={cx - 26} y={y(p.exigido) - 8} width={52} height={16} fill="transparent"
                    data-tip={`${rot(p.c.cycle_id)} · F1 exigido ${f4(p.exigido)}`} />
                  <circle cx={cxv} cy={y(p.vig)} r={6} fill="var(--color-modelo-vigente)" stroke="var(--color-surface)" strokeWidth={2} />
                  <circle cx={cxv} cy={y(p.vig)} r={13} fill="transparent"
                    data-tip={`${rot(p.c.cycle_id)} · vigente reavaliado (${nomeModelo(ant)}, ${rot(ant.cycle_id)}): F1 ${f4(p.vig)}`} />
                </>
              ) : (
                <text x={cx} y={mt + 10} textAnchor="middle" className="muted">sem vigente · exigido {f2(p.exigido)}</text>
              )}
              <circle cx={cx} cy={y(p.venc)} r={6} fill={cor} stroke="var(--color-surface)" strokeWidth={2} />
              <circle cx={cx} cy={y(p.venc)} r={13} fill="transparent"
                data-tip={`${rot(p.c.cycle_id)} · vencedor ${nomeModelo(p.c.vencedor)}: F1 ${f4(p.venc)} → ${p.c.decisao}`} />
              <text className="rotulo-valor" x={cx + 12} y={y(p.venc) + 4}>{f4(p.venc)}</text>
              <text x={xc(i)} y={H - mb + 18} textAnchor="middle" style={{ fill: 'var(--color-text)', fontWeight: 600 }}>
                {rot(p.c.cycle_id)}
              </text>
              <text x={xc(i)} y={H - mb + 34} textAnchor="middle">{p.c.decisao}</text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// --------------------------------------------------------------------------
// Barras agrupadas: precision, recall e F1 de A, B e vigente
// --------------------------------------------------------------------------
export function GraficoBarras({ c, metricas }: { c: CompeticaoCiclo; metricas: Metrica[] }) {
  const [ref, largura] = useLargura<HTMLDivElement>()
  const W = largura, H = 260, ml = 40, mr = 8, mt = 16, mb = 28
  const n = c.modelos.length, grupo = (W - ml - mr) / metricas.length
  const bw = Math.min(56, (grupo * 0.7 - (n - 1) * 2) / n)
  const y = (v: number) => mt + (1 - v) * (H - mt - mb)
  const barra = (x: number, v: number) => {
    const top = y(v), base = y(0), r = Math.min(4, bw / 2, base - top)
    return `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}H${x + bw - r}Q${x + bw},${top} ${x + bw},${top + r}V${base}Z`
  }
  return (
    <div ref={ref}>
      <div className="legenda">
        {c.modelos.map((m) => (
          <span key={m.papel}><Ponto papel={m.papel} />{PAPEL_NOME[m.papel]} · {ALG[m.algoritmo]}</span>
        ))}
      </div>
      <svg className="grafico" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Barras agrupadas de precision, recall e F1 por modelo">
        {[0, 0.25, 0.5, 0.75, 1].map((t) => (
          <g key={t}>
            <line className="grade-linha" x1={ml} x2={W - mr} y1={y(t)} y2={y(t)} />
            <text x={ml - 6} y={y(t) + 4} textAnchor="end">{f2(t)}</text>
          </g>
        ))}
        {metricas.map((k, gi) => {
          const x0 = ml + gi * grupo + (grupo - (n * bw + (n - 1) * 2)) / 2
          return (
            <g key={k}>
              {c.modelos.map((m, i) => {
                const x = x0 + i * (bw + 2), v = m.metricas[k]
                return (
                  <g key={m.papel}>
                    <path d={barra(x, v)} fill={`var(--color-modelo-${m.papel.toLowerCase()})`}
                      data-tip={`${PAPEL_NOME[m.papel]} (${ALG[m.algoritmo]}) · ${METRICA[k]} ${f4(v)}`} />
                    {k === 'f1' && bw >= 36 && (
                      <text className="rotulo-valor" x={x + bw / 2} y={y(v) - 6} textAnchor="middle" style={{ fontSize: 11 }}>
                        {f4(v).replace(/^0/, '')}
                      </text>
                    )}
                  </g>
                )
              })}
              <text x={ml + gi * grupo + grupo / 2} y={H - 8} textAnchor="middle" style={{ fill: 'var(--color-text)', fontWeight: 600 }}>
                {METRICA[k]}
              </text>
            </g>
          )
        })}
        <line className="eixo-forte" x1={ml} x2={W - mr} y1={y(0)} y2={y(0)} />
      </svg>
    </div>
  )
}
