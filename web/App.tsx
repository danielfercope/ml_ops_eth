// Layout, navegação e rotas do painel (Dev 5 · critério 10).
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, Navigate, NavLink, Route, Routes, useLocation } from 'react-router'
import { CENARIOS, useDadosPainel, type Cenario } from './dados/carregar'
import { PainelContexto } from './dados/contexto'
import { ARQUIVOS_PAINEL } from './dados/tipos'
import { Carregando, ErroCarregamento, Icone, Tooltip } from './componentes'
import { fmtDataLonga, rotuladorCiclos } from './formato'
import Ciclos from './telas/Ciclos'
import Competicao from './telas/Competicao'
import Previsoes from './telas/Previsoes'
import Votacao from './telas/Votacao'
import VisaoGeral from './telas/VisaoGeral'

const ROTAS = [
  { caminho: 'visao-geral', nome: 'Visão geral', Tela: VisaoGeral },
  { caminho: 'ciclos', nome: 'Ciclos', Tela: Ciclos },
  { caminho: 'votacao', nome: 'Votação', Tela: Votacao },
  { caminho: 'competicao', nome: 'Competição', Tela: Competicao },
  { caminho: 'previsoes', nome: 'Previsões', Tela: Previsoes },
] as const

type Tema = 'auto' | 'light' | 'dark'
const TEMAS: Tema[] = ['auto', 'light', 'dark']
const NOME_TEMA: Record<Tema, string> = { auto: 'auto', light: 'claro', dark: 'escuro' }

function lerTema(): Tema {
  const daUrl = new URLSearchParams(window.location.search).get('tema') // ?tema=dark para prints do relatório
  let salvo: string | null = null
  try {
    salvo = localStorage.getItem('painel-tema')
  } catch {
    /* sem storage */
  }
  const t = (daUrl ?? salvo ?? 'auto') as Tema
  return TEMAS.includes(t) ? t : 'auto'
}

function useTema() {
  const [tema, setTema] = useState<Tema>(lerTema)
  useEffect(() => {
    if (tema === 'auto') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.setAttribute('data-theme', tema)
    try {
      localStorage.setItem('painel-tema', tema)
    } catch {
      /* sem storage */
    }
  }, [tema])
  return [tema, () => setTema((t) => TEMAS[(TEMAS.indexOf(t) + 1) % TEMAS.length])] as const
}

export default function App({ cenarioInicial = 'pipeline', pedido = false }: { cenarioInicial?: Cenario; pedido?: boolean }) {
  const { cenario, escolher, estado, carregarArquivos, tentarDeNovo } = useDadosPainel(cenarioInicial, pedido)
  const [tema, alternarTema] = useTema()
  const arquivos = useRef<HTMLInputElement>(null)
  const { pathname } = useLocation()
  const rota = ROTAS.find((r) => pathname === `/${r.caminho}`) ?? ROTAS[0]

  useEffect(() => {
    document.title = `${rota.nome} · Painel MLOps · Fraude Ethereum`
    window.scrollTo?.(0, 0)
  }, [rota])

  const painel = useMemo(
    () => (estado.status === 'ok' ? { dados: estado.dados, rotCiclo: rotuladorCiclos(estado.dados.ciclos.ciclos) } : null),
    [estado],
  )

  const trocarCenario = (c: Cenario) => {
    escolher(c)
    const u = new URL(window.location.href)
    u.searchParams.set('cenario', c)
    window.history.replaceState(window.history.state, '', u)
  }

  let conteudo
  if (estado.status === 'carregando') conteudo = <Carregando />
  else if (estado.status === 'erro')
    conteudo = <ErroCarregamento erro={estado.erro} onTentar={tentarDeNovo} onArquivos={() => arquivos.current?.click()} />
  else {
    const mock = ARQUIVOS_PAINEL.some((n) => estado.dados[n].origem === 'mock')
    conteudo = (
      <PainelContexto.Provider value={painel}>
        {mock && (
          <div className="aviso aviso--mock" role="note">
            <Icone nome="info" />
            <div>
              <b>Dados fictícios.</b> Cenário mock “{cenario}”: <span className="mono">cycle_id</span>, datas,{' '}
              <span className="mono">run_id</span>, links e endereços são inventados; as métricas oficiais vêm do pipeline.
            </div>
          </div>
        )}
        <Routes>
          {ROTAS.map(({ caminho, Tela }) => <Route key={caminho} path={`/${caminho}`} element={<Tela />} />)}
          <Route path="*" element={<Navigate to="/visao-geral" replace />} />
        </Routes>
      </PainelContexto.Provider>
    )
  }

  return (
    <>
      <header className="topo">
        <div className="wrap">
          <div className="topo__linha">
            <Link className="marca" to="/visao-geral" aria-label="Painel MLOps · Fraude Ethereum: ir para a visão geral">
              <svg className="marca__icone" width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
                <rect width="32" height="32" rx="8" fill="var(--color-text)" />
                <path d="M16 5l-6.5 11L16 20l6.5-4L16 5z" fill="var(--color-surface)" />
                <path d="M9.5 17.5L16 27l6.5-9.5L16 21.5l-6.5-4z" fill="var(--color-surface)" opacity=".65" />
              </svg>
              <span className="marca__texto">
                <b>Painel MLOps · Fraude Ethereum</b>
                <span>Ciclos do pipeline, votação de features, competição A/B e previsões</span>
              </span>
            </Link>
            <div className="grupo-mock" role="group" aria-label="Fonte dos dados">
              <span className="grupo-mock__rotulo">Dados</span>
              <label className="controle">
                Cenário
                <select value={cenario} onChange={(e) => trocarCenario(e.target.value as Cenario)}>
                  {CENARIOS.map((c) => <option key={c.id} value={c.id}>{c.nome}</option>)}
                </select>
              </label>
            </div>
            <button className="botao" type="button" title="Alternar tema" onClick={alternarTema}>
              Tema: {NOME_TEMA[tema]}
            </button>
          </div>
          <nav className="nav" aria-label="Telas">
            {ROTAS.map((r) => <NavLink key={r.caminho} to={`/${r.caminho}`}>{r.nome}</NavLink>)}
          </nav>
        </div>
      </header>

      <main className="wrap" aria-live="polite">{conteudo}</main>

      <footer className="rodape">
        {estado.status === 'ok' && (
          <div className="wrap">
            <span>Fonte: <span className="mono">{estado.fonte}</span></span>
            <span>
              gerado em {fmtDataLonga(estado.dados.ciclos.gerado_em)} · versao_formato {estado.dados.ciclos.versao_formato} ·
              origem {estado.dados.ciclos.origem}
            </span>
          </div>
        )}
        <div className="wrap creditos">
          <span>Trabalho N3 · MLOps · recurso adicional (critério 10)</span>
          <span>
            Dados:{' '}
            <a href="https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset" target="_blank" rel="noopener">
              Ethereum Fraud Detection Dataset
            </a>{' '}
            (Kaggle, ODbL)
          </span>
          <span><a href="https://github.com/danielfercope/ml_ops_eth" target="_blank" rel="noopener">Código no GitHub</a></span>
        </div>
      </footer>
      <Tooltip />
      <input type="file" ref={arquivos} accept=".json,application/json" multiple hidden
        onChange={(e) => {
          if (e.target.files?.length) carregarArquivos(e.target.files)
          e.target.value = ''
        }} />
    </>
  )
}
