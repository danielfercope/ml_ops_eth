"""Gera o relatório técnico do N3 (``reports/relatorio_n3.pdf``) a partir dos docs.

Cada frente mantém a sua seção em ``docs/*.md``; este script junta tudo, na
ordem dos critérios, em um HTML único (``reports/relatorio_n3.html``) e o
imprime em PDF com o Chrome/Chromium em modo headless.

Uso (da raiz do repositório):
    pip install markdown            # só para gerar o relatório
    python reports/gerar_relatorio.py
    python reports/gerar_relatorio.py --sem-pdf     # só o HTML

Rode de novo depois de cada ciclo relevante para atualizar os números.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DIR = RAIZ / "reports"

# ordem das seções = ordem dos critérios de avaliação
SECOES = [
    "reports/capa.md",
    "docs/dados.md",                 # critérios 1 e 2
    "docs/votacao_features.md",      # critérios 3 e 4
    "docs/competicao_promocao.md",   # critérios 5 e 8
    "docs/arquitetura_fluxo.md",     # critérios 6, 7, 8 e 9
    "docs/recurso_adicional.md",     # critério 10
]

CHROMES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
]

CSS = """
@page { size: A4; margin: 18mm 16mm 20mm; }
body { font: 10.5pt/1.5 -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; }
h1 { font-size: 19pt; margin: 0 0 8pt; page-break-before: always; border-bottom: 2px solid #256abf; padding-bottom: 4pt; }
section:first-of-type h1 { page-break-before: avoid; font-size: 22pt; }
h2 { font-size: 14pt; margin: 16pt 0 6pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; }
h2, h3 { page-break-after: avoid; }
p, li { orphans: 3; widows: 3; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9c8c1; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f0efec; }
code { font: 8.8pt ui-monospace, Menlo, Consolas, monospace; background: #f0efec; padding: 0 2pt; border-radius: 2pt; overflow-wrap: anywhere; }
pre { background: #f4f3f0; border: 1px solid #e3e2dd; padding: 6pt 8pt; border-radius: 4pt; font-size: 8.5pt;
      white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 6pt 0; padding: 2pt 10pt; border-left: 3px solid #86b6ef; color: #52514e; }
a { color: #256abf; text-decoration: none; }
.toc { page-break-after: always; }
.toc ul { list-style: none; padding-left: 12pt; }
.toc > ul { padding-left: 0; }
.rodape { color: #75746e; font-size: 8.5pt; margin-top: 20pt; }
"""


def _preparar(md: str) -> str:
    """Ajustes para o documento único: tira as notas internas de cada doc
    ('> Base para a seção ...') e transforma links relativos em texto."""
    linhas = md.splitlines()
    out, pulando = [], False
    for i, l in enumerate(linhas):
        if l.startswith("> Base para") or l.startswith("> Tarefa 1 do Dev 5"):
            pulando = True
        if pulando:
            if l.startswith(">"):
                continue
            pulando = False
        out.append(l)
    texto = "\n".join(out)
    return re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]+\)", r"\1", texto)


def montar_html() -> str:
    import markdown

    partes = []
    for rel in SECOES:
        md = _preparar((RAIZ / rel).read_text(encoding="utf-8"))
        corpo = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists", "toc"],
                                  extension_configs={"toc": {"permalink": False}})
        partes.append(f"<section>{corpo}</section>")

    # sumário com os títulos de nível 1 e 2 de todas as seções
    html = "\n".join(partes)
    itens, atual = [], None
    for nivel, ident, titulo in re.findall(r'<h([12]) id="([^"]+)">(.*?)</h\1>', html):
        if nivel == "1":
            atual = [titulo, ident, []]
            itens.append(atual)
        elif atual is not None:
            atual[2].append((titulo, ident))
    toc = "<nav class='toc'><h2>Sumário</h2><ul>" + "".join(
        f"<li><a href='#{i}'><b>{t}</b></a><ul>" + "".join(f"<li><a href='#{si}'>{st}</a></li>" for st, si in sub)
        + "</ul></li>" for t, i, sub in itens[1:]) + "</ul></nav>"
    capa, resto = partes[0], "\n".join(partes[1:])
    gerado = datetime.now().strftime("%d/%m/%Y %H:%M")
    return (f"<!doctype html><html lang='pt-BR'><head><meta charset='utf-8'><title>Relatório N3 — MLOps</title>"
            f"<style>{CSS}</style></head><body>{capa}<p class='rodape'>Gerado em {gerado} a partir de docs/*.md "
            f"por reports/gerar_relatorio.py.</p>{toc}{resto}</body></html>")


def _chrome() -> str | None:
    for c in CHROMES:
        if Path(c).exists() or shutil.which(c):
            return c
    return None


def imprimir_pdf(html: Path, pdf: Path) -> None:
    chrome = _chrome()
    if not chrome:
        raise RuntimeError("Chrome/Chromium não encontrado; abra o HTML e imprima em PDF manualmente")
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", html.as_uri()],
                   check=True, capture_output=True, timeout=120)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gera reports/relatorio_n3.{html,pdf}")
    ap.add_argument("--sem-pdf", action="store_true", help="gera só o HTML")
    a = ap.parse_args()
    html = DIR / "relatorio_n3.html"
    html.write_text(montar_html(), encoding="utf-8")
    print(f"HTML: {html.relative_to(RAIZ)}")
    if not a.sem_pdf:
        pdf = DIR / "relatorio_n3.pdf"
        try:
            imprimir_pdf(html, pdf)
        except (RuntimeError, subprocess.SubprocessError) as e:
            print(f"PDF não gerado: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"PDF:  {pdf.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
