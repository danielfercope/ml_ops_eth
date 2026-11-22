import react from '@vitejs/plugin-react'
import { cpSync } from 'node:fs'
import { resolve } from 'node:path'
import { defineConfig, type Plugin } from 'vite'

// O painel (index.html) lê os mocks de web/mocks/<cenario>/. No `vite dev` eles
// já são servidos da raiz; no build, copiamos a pasta para dist/ para o site
// publicado funcionar igual. Sai quando o painel passar a ler public/data/.
function copiarMocks(): Plugin {
  return {
    name: 'copiar-mocks-painel',
    apply: 'build',
    closeBundle() {
      cpSync(resolve(import.meta.dirname, 'web/mocks'), resolve(import.meta.dirname, 'dist/web/mocks'), { recursive: true })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), copiarMocks()],
})
