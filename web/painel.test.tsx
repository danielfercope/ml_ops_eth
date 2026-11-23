// Testes de renderização do painel com os JSONs mock (Dev 5, tarefa 7).
// Cada tela é renderizada em cada cenário (padrao, mantido, vazio) com o fetch
// servindo os mocks versionados em web/dados/mocks/.
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { aceitar, buscarPainel, type Cenario } from './dados/carregar'
import { ARQUIVOS_PAINEL } from './dados/tipos'

const DIR_MOCKS = resolve(import.meta.dirname, 'dados/mocks')
const lerMock = (cenario: string, nome: string) =>
  JSON.parse(readFileSync(resolve(DIR_MOCKS, cenario, `${nome}.json`), 'utf-8'))

/** fetch falso: mocks em web/dados/mocks/<c>/; "data/" (pipeline) responde 404, salvo se `pipeline` for dado. */
function fetchFalso(pipeline?: string) {
  return vi.fn(async (url: string) => {
    const m = url.match(/web\/dados\/mocks\/(\w+)\/(\w+)\.json$/) ?? url.match(/()data\/(\w+)\.json$/)
    if (!m) return new Response(null, { status: 404 })
    const cenario = m[1] || pipeline
    if (!cenario) return new Response(null, { status: 404 })
    return new Response(JSON.stringify(lerMock(cenario, m[2])), { status: 200 })
  })
}

function renderizar(rota: string, cenario: Cenario = 'padrao', pedido = true) {
  return render(
    <MemoryRouter initialEntries={[rota]}>
      <App cenarioInicial={cenario} pedido={pedido} />
    </MemoryRouter>,
  )
}
const principal = () => within(screen.getByRole('main'))
const titulo = async (texto: string) => principal().findByRole('heading', { level: 1, name: texto })

beforeEach(() => {
  vi.stubGlobal('fetch', fetchFalso())
  window.scrollTo = vi.fn() as unknown as typeof window.scrollTo // jsdom não implementa
})
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

describe('carregamento', () => {
  it('aceita os cinco arquivos e recusa versao_formato desconhecida', () => {
    const d = Object.fromEntries(ARQUIVOS_PAINEL.map((n) => [n, lerMock('padrao', n)]))
    expect(aceitar(d).vigente.modelo_mudou).toBe(true)
    expect(() => aceitar({ ...d, votacao: { ...d.votacao, versao_formato: 2 } })).toThrow(/versao_formato/)
    expect(() => aceitar({ ...d, ciclos: undefined })).toThrow(/falta ciclos.json/)
  })

  it('busca os dados reais em data/', async () => {
    const f = fetchFalso('mantido')
    const d = await buscarPainel('pipeline', f as unknown as typeof fetch)
    expect(f.mock.calls[0][0]).toMatch(/\/data\/ciclos\.json$/)
    expect(d.ciclos.ciclos).toHaveLength(3)
  })

  it('sem dados reais e sem cenário pedido, cai no mock padrão', async () => {
    renderizar('/visao-geral', 'pipeline', false)
    await titulo('Visão geral')
    expect(screen.getByRole('note').textContent).toMatch(/Cenário mock “padrao”/)
  })

  it('mostra o erro quando o cenário pedido não carrega', async () => {
    renderizar('/visao-geral', 'pipeline', true)
    const alerta = await screen.findByRole('alert')
    expect(alerta.textContent).toMatch(/HTTP 404/)
    expect(within(alerta).getByRole('button', { name: 'Tentar de novo' })).toBeTruthy()
  })

  it('rota desconhecida vai para a visão geral', async () => {
    renderizar('/nao-existe')
    await titulo('Visão geral')
  })
})

describe('Visão geral', () => {
  it('padrão: banner de troca, KPIs do vigente e selo da última decisão', async () => {
    renderizar('/visao-geral')
    await titulo('Visão geral')
    const banner = screen.getByRole('status')
    expect(banner.textContent).toMatch(/O modelo vigente mudou no último ciclo/)
    expect(banner.textContent).toMatch(/XGBoost \(B\) · Ciclo 2/)
    expect(principal().getAllByText('0,9429').length).toBeGreaterThan(0)
    expect(principal().getAllByText('Promovido').length).toBeGreaterThan(0)
  })

  it('mostra o resultado das verificações de qualidade', async () => {
    renderizar('/visao-geral')
    await titulo('Visão geral')
    expect(principal().getByText('vencedor aprovado nas verificações')).toBeTruthy()
  })

  it('mantido: sem banner e com "sem dados novos"', async () => {
    renderizar('/visao-geral', 'mantido')
    await titulo('Visão geral')
    expect(screen.queryByRole('status')).toBeNull()
    expect(principal().getByText(/sem dados novos/)).toBeTruthy()
  })

  it('vazio: estado vazio e regra de promoção', async () => {
    renderizar('/visao-geral', 'vazio')
    await titulo('Visão geral')
    expect(principal().getByText('Nenhum ciclo executado ainda')).toBeTruthy()
    expect(principal().getByText('Regra de promoção')).toBeTruthy()
  })
})

describe('Ciclos', () => {
  it('uma linha por ciclo, do mais recente para o mais antigo', async () => {
    renderizar('/ciclos', 'mantido')
    await titulo('Ciclos')
    const linhas = principal().getAllByRole('row').slice(1)
    expect(linhas).toHaveLength(3)
    expect(linhas[0].textContent).toMatch(/^Ciclo 3/)
    expect(within(linhas[0]).getByText('Mantido')).toBeTruthy()
    expect(within(linhas[2]).getByText('Promovido')).toBeTruthy()
  })

  it('vazio', async () => {
    renderizar('/ciclos', 'vazio')
    expect(await principal().findByText('Nenhum ciclo ainda')).toBeTruthy()
  })
})

describe('Votação', () => {
  it('tabela de votos com as eleitas destacadas e troca de ciclo', async () => {
    renderizar('/votacao')
    await titulo('Votação de features')
    const vot = lerMock('padrao', 'votacao')
    const c2 = vot.ciclos[1]
    const eleitas = principal().getAllByRole('row').filter((r) => r.classList.contains('eleita'))
    expect(eleitas).toHaveLength(c2.eleitas.length)
    expect(principal().getByText(/Em relação ao Ciclo 1:/)).toBeTruthy()

    fireEvent.click(principal().getByRole('button', { name: 'Ciclo 1' }))
    const c1 = vot.ciclos[0]
    await waitFor(() =>
      expect(principal().getAllByRole('row').filter((r) => r.classList.contains('eleita'))).toHaveLength(c1.eleitas.length),
    )
    expect(principal().queryByText(/Em relação ao/)).toBeNull()
  })

  it('ocultar features sem voto', async () => {
    renderizar('/votacao')
    await titulo('Votação de features')
    const c2 = lerMock('padrao', 'votacao').ciclos[1]
    fireEvent.click(principal().getByRole('checkbox'))
    const com = c2.tabela.filter((l: { total_votos: number }) => l.total_votos > 0).length
    expect(principal().getAllByRole('row').length - 1).toBe(com)
  })
})

describe('Competição', () => {
  it('barras de A, B e vigente e uma matriz por modelo', async () => {
    renderizar('/competicao')
    await titulo('Competição')
    expect(principal().getByRole('img', { name: /Barras agrupadas/ })).toBeTruthy()
    expect(principal().getAllByRole('table', { name: 'Matriz de confusão' })).toHaveLength(3)
    expect(principal().getAllByText('Vigente reavaliado').length).toBeGreaterThan(0)
  })

  it('ciclo 1 sem vigente: só A e B', async () => {
    renderizar('/competicao')
    await titulo('Competição')
    fireEvent.click(principal().getByRole('button', { name: 'Ciclo 1' }))
    await waitFor(() => expect(principal().getAllByRole('table', { name: 'Matriz de confusão' })).toHaveLength(2))
  })
})

describe('Previsões', () => {
  it('amostra com probabilidade, previsão, rótulo real e filtro por resultado', async () => {
    renderizar('/previsoes')
    await titulo('Previsões')
    const c = lerMock('padrao', 'previsoes').ciclos.at(-1)
    const amostra = () => principal().getAllByRole('row').filter((r) => r.querySelector('.prob'))
    expect(amostra()).toHaveLength(c.amostra.length)

    const fn = c.amostra.filter((l: { previsao: number; rotulo_real: number }) => !l.previsao && l.rotulo_real).length
    fireEvent.click(principal().getByRole('button', { name: new RegExp(`^FN ${fn}$`) }))
    expect(amostra()).toHaveLength(fn)
  })

  it('vazio', async () => {
    renderizar('/previsoes', 'vazio')
    expect(await principal().findByText('Sem previsões ainda')).toBeTruthy()
  })
})

describe('navegação', () => {
  it('links para as 5 telas, com a atual marcada', async () => {
    renderizar('/ciclos')
    await titulo('Ciclos')
    const nav = within(screen.getByRole('navigation', { name: 'Telas' }))
    expect(nav.getAllByRole('link').map((a) => a.textContent)).toEqual(
      ['Visão geral', 'Ciclos', 'Votação', 'Competição', 'Previsões'])
    expect(nav.getByRole('link', { name: 'Ciclos' }).getAttribute('aria-current')).toBe('page')
    fireEvent.click(nav.getByRole('link', { name: 'Previsões' }))
    await titulo('Previsões')
  })
})
