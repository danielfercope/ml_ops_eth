import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router'
import App from './App.tsx'
import { CENARIOS, type Cenario } from './dados/carregar'
import './estilos.css'

// ?cenario=pipeline|padrao|mantido|vazio escolhe a fonte; sem ele, dados reais com fallback para o mock.
// HashRouter: as rotas (#/ciclos, ...) funcionam no GitHub Pages sem configuração de servidor.
const pedido = new URLSearchParams(window.location.search).get('cenario')
const valido = CENARIOS.some((c) => c.id === pedido)

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <App cenarioInicial={valido ? (pedido as Cenario) : 'pipeline'} pedido={valido} />
    </HashRouter>
  </StrictMode>,
)
