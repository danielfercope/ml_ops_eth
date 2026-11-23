// Carrega os cinco JSONs do painel (contrato em docs/painel_dados.md).
import { useCallback, useEffect, useState } from 'react'
import { ARQUIVOS_PAINEL, type ArquivosPainel } from './tipos'

/** "pipeline" = dados reais em data/ (public/data/); os demais são mocks versionados. */
export type Cenario = 'pipeline' | 'padrao' | 'mantido' | 'vazio'
export const CENARIOS: { id: Cenario; nome: string }[] = [
  { id: 'pipeline', nome: 'pipeline (dados reais)' },
  { id: 'padrao', nome: 'padrão (2 ciclos, troca)' },
  { id: 'mantido', nome: 'mantido (3º ciclo sem troca)' },
  { id: 'vazio', nome: 'vazio (antes do 1º ciclo)' },
]
const VERSAO_FORMATO = 1

export function urlBase(cenario: Cenario, base = import.meta.env.BASE_URL): string {
  return cenario === 'pipeline' ? `${base}data/` : `${base}web/dados/mocks/${cenario}/`
}

export function fonteLegivel(cenario: Cenario): string {
  return cenario === 'pipeline' ? 'public/data/' : `web/dados/mocks/${cenario}/`
}

/** Confere presença e versão dos cinco arquivos. */
export function aceitar(d: Partial<Record<string, unknown>>): ArquivosPainel {
  for (const n of ARQUIVOS_PAINEL) {
    const arq = d[n] as { versao_formato?: number } | undefined
    if (!arq) throw new Error(`falta ${n}.json`)
    if (arq.versao_formato !== VERSAO_FORMATO)
      throw new Error(`${n}.json: versao_formato ${arq.versao_formato}, o painel entende ${VERSAO_FORMATO}`)
  }
  return d as unknown as ArquivosPainel
}

export async function buscarPainel(cenario: Cenario, fetcher: typeof fetch = fetch): Promise<ArquivosPainel> {
  const base = urlBase(cenario)
  const pares = await Promise.all(
    ARQUIVOS_PAINEL.map(async (nome) => {
      const url = `${base}${nome}.json`
      let r: Response
      try {
        r = await fetcher(url, { cache: 'no-store' })
      } catch (e) {
        throw new Error(`${nome}.json: não foi possível buscar ${url} (${(e as Error).message})`, { cause: e })
      }
      if (!r.ok) throw new Error(`${nome}.json: HTTP ${r.status} em ${url}`)
      return [nome, await r.json()] as const
    }),
  )
  return aceitar(Object.fromEntries(pares))
}

export type EstadoPainel =
  | { status: 'carregando' }
  | { status: 'erro'; erro: string }
  | { status: 'ok'; dados: ArquivosPainel; fonte: string }

/**
 * Busca os dados do cenário. Se o cenário não foi pedido explicitamente e os
 * dados reais ainda não existem (antes do 1º export), cai no mock "padrao".
 */
export function useDadosPainel(cenarioInicial: Cenario, pedido: boolean) {
  const [cenario, setCenario] = useState<Cenario>(cenarioInicial)
  const [explicito, setExplicito] = useState(pedido)
  const [estado, setEstado] = useState<EstadoPainel>({ status: 'carregando' })
  const [tentativa, setTentativa] = useState(0)

  useEffect(() => {
    let vivo = true
    buscarPainel(cenario)
      .then((dados) => vivo && setEstado({ status: 'ok', dados, fonte: fonteLegivel(cenario) }))
      .catch((e: Error) => {
        if (!vivo) return
        if (cenario === 'pipeline' && !explicito) setCenario('padrao')
        else setEstado({ status: 'erro', erro: e.message })
      })
    return () => {
      vivo = false
    }
  }, [cenario, explicito, tentativa])

  // o estado "carregando" é marcado aqui (evento), não dentro do efeito
  const escolher = useCallback((c: Cenario) => {
    setEstado({ status: 'carregando' })
    setExplicito(true)
    setCenario(c)
  }, [])
  const carregarArquivos = useCallback(async (files: FileList) => {
    try {
      const d: Record<string, unknown> = {}
      for (const f of Array.from(files)) d[f.name.replace(/\.json$/, '')] = JSON.parse(await f.text())
      const nomes = Array.from(files).map((f) => f.name).join(', ')
      setEstado({ status: 'ok', dados: aceitar(d), fonte: `arquivos locais (${nomes})` })
    } catch (e) {
      setEstado({ status: 'erro', erro: (e as Error).message })
    }
  }, [])
  const tentarDeNovo = useCallback(() => {
    setEstado({ status: 'carregando' })
    setTentativa((t) => t + 1)
  }, [])

  return { cenario, escolher, estado, carregarArquivos, tentarDeNovo }
}
