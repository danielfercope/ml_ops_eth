/// <reference types="vitest/config" />
import react from '@vitejs/plugin-react'
import { cpSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vite'

const raiz = import.meta.dirname

// O seletor "Cenário" do painel e o protótipo leem os mocks de web/dados/mocks/<cenario>/.
// No `vite dev` eles já são servidos da raiz; no build, copiamos a pasta para dist/.
// Os dados reais (public/data/) entram no build pelo publicDir padrão.
function copiarMocks(): Plugin {
  return {
    name: 'copiar-mocks-painel',
    apply: 'build',
    closeBundle() {
      cpSync(resolve(raiz, 'web/dados/mocks'), resolve(raiz, 'dist/web/dados/mocks'), { recursive: true })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), copiarMocks()],
  build: {
    rollupOptions: {
      // index.html = app React; prototipo.html = protótipo aprovado (HTML único), publicado junto
      input: { main: resolve(raiz, 'index.html'), prototipo: resolve(raiz, 'prototipo.html') },
    },
  },
  test: {
    environment: 'jsdom',
    include: ['web/**/*.test.{ts,tsx}'],
  },
})
