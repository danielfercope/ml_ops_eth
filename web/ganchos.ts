// Hooks compartilhados pelas telas e gráficos.
import { useLayoutEffect, useRef, useState } from 'react'

/** Ciclo escolhido numa tela; padrão = o último da lista. */
export function useCicloEscolhido(ids: string[]): [string, (id: string) => void] {
  const [escolhido, setEscolhido] = useState<string | null>(null)
  const atual = escolhido && ids.includes(escolhido) ? escolhido : ids[ids.length - 1]
  return [atual, setEscolhido]
}

/** Largura interna do elemento, para desenhar o SVG em 1:1 (texto sem escalar). */
export function useLargura<T extends HTMLElement>(minimo = 300, padrao = 900) {
  const ref = useRef<T>(null)
  const [largura, setLargura] = useState(padrao)
  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const medir = () => el.clientWidth > 0 && setLargura(Math.max(minimo, el.clientWidth))
    medir()
    if (typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(medir)
    ro.observe(el)
    return () => ro.disconnect()
  }, [minimo])
  return [ref, largura] as const
}
