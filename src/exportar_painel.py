"""Exporta os JSONs do painel a partir do estado do pipeline (Dev 5, critério 10).

Lê o que os notebooks 01–05 deixam commitado e grava os cinco arquivos do
contrato (``docs/painel_dados.md``) em ``public/data/``:

====================  =======================================================
Arquivo               Fontes
====================  =======================================================
``ciclos.json``       ``models/historico_decisoes.csv`` (ordem dos ciclos),
                      ``artifacts/<cycle_id>/decisao.json``, métricas do run
                      ``etl`` no MLflow e ``evidencias/<cycle_id>/resumo_ciclo.json``
``votacao.json``      ``artifacts/<cycle_id>/tabela_votos.csv``,
                      ``resumo_votacao.json`` e ``features_eleitas.json``
``competicao.json``   ``artifacts/<cycle_id>/competicao.json`` (A e B) e
                      ``decisao.json`` (vigente reavaliado na validação)
``vigente.json``      ``models/modelo_vigente.json`` + último ciclo
``previsoes.json``    ``artifacts/<cycle_id>/previsoes.csv`` + lotes em ``data/``
====================  =======================================================

Antes de gravar, valida tudo com ``src.contrato_painel`` (schema + coerência).
Se algo não bate, nada é escrito e o processo sai com erro; o workflow trata
esse passo com ``continue-on-error``, então o painel nunca bloqueia o ciclo.

Uso:
    python -m src.exportar_painel                  # grava em public/data/
    python -m src.exportar_painel --saida /tmp/x   # outra pasta
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.contrato_painel import ARQUIVOS, validar_coerencia, validar_formato
from src.ingestao import RAIZ, carregar_config, ler_registro
from src.rastreio import caminho_db

SAIDA_PADRAO = RAIZ / "public" / "data"
VERSAO_FORMATO = 1
METRICAS = ["precision", "recall", "f1"]
NOME_METODO = {
    "selectkbest_f_classif": "SelectKBest (F)",
    "selectkbest_mutual_info": "SelectKBest (MI)",
    "random_forest": "Random Forest",
    "rfe_logistic": "RFE (Logística)",
}
# amostra da tela Previsões: por tipo de resultado, sorteada com seed fixa
AMOSTRA_POR_TIPO = {"VP": 8, "VN": 8, "FP": 4, "FN": 4}
SEED_AMOSTRA = 42


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _r4(x) -> float:
    return round(float(x), 4)


def _iso_z(texto: str | None) -> str | None:
    """'2026-11-21T13:12:00+00:00' → '2026-11-21T13:12:00Z' (UTC, sem frações)."""
    if not texto:
        return None
    dt = datetime.fromisoformat(str(texto).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _agora_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ler_json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _matriz(m: dict) -> dict:
    return {"vn": int(m["tn"]), "fp": int(m["fp"]), "fn": int(m["fn"]), "vp": int(m["tp"])}


def _metricas(m: dict) -> dict:
    return {k: _r4(m[k]) for k in METRICAS}


def _ref(cycle_id: str, modelo: str, algoritmo: str, run_id: str) -> dict:
    return {"cycle_id": cycle_id, "modelo": modelo, "algoritmo": algoritmo, "run_id": run_id}


SEM_LOTES = "nenhum"   # o notebook 01 registra lotes_novos="nenhum" num ciclo sem dados novos


def _lista(valor) -> list[str]:
    if valor is None or (isinstance(valor, float) and math.isnan(valor)):
        return []
    return [x for x in str(valor).split(",") if x and x != SEM_LOTES]


# --------------------------------------------------------------------------- #
# Leitura das fontes
# --------------------------------------------------------------------------- #
class Fontes:
    """Acesso às fontes de um repositório (raiz configurável para testes)."""

    def __init__(self, raiz: Path = RAIZ, cfg: dict | None = None):
        self.raiz = Path(raiz)
        self.cfg = cfg or carregar_config()
        self.artifacts = self.raiz / self.cfg.get("votacao", {}).get("dir_artifacts", "artifacts")
        self.evidencias = self.raiz / "evidencias"

    def _na_raiz(self, p: str | Path) -> Path:
        """Caminho do config (relativo ao repositório) reancorado em ``self.raiz``."""
        p = Path(p)
        if p.is_absolute():
            p = p.relative_to(RAIZ) if p.is_relative_to(RAIZ) else p
        return p if p.is_absolute() else self.raiz / p

    def historico(self) -> pd.DataFrame:
        p = self._na_raiz(self.cfg.get("promocao", {}).get("historico", "models/historico_decisoes.csv"))
        return pd.read_csv(p, dtype=str) if p.exists() else pd.DataFrame()

    def manifesto(self) -> dict | None:
        return _ler_json(self._na_raiz(self.cfg.get("promocao", {}).get("manifesto", "models/modelo_vigente.json")))

    def artefato_json(self, cycle_id: str, nome: str) -> dict | None:
        return _ler_json(self.artifacts / cycle_id / nome)

    def artefato_csv(self, cycle_id: str, nome: str) -> pd.DataFrame | None:
        p = self.artifacts / cycle_id / nome
        return pd.read_csv(p) if p.exists() else None

    def resumo_ciclo(self, cycle_id: str) -> dict | None:
        return _ler_json(self.evidencias / cycle_id / "resumo_ciclo.json")

    def run_etl(self, cycle_id: str) -> dict:
        """Métricas e params do run ``etl`` do ciclo (tabelas do MLflow em SQLite)."""
        db = self._na_raiz(caminho_db(self.cfg))
        if not db.exists():
            return {}
        with sqlite3.connect(db) as con:
            q = ("select r.run_uuid from runs r "
                 "join tags c on c.run_uuid = r.run_uuid and c.key = 'cycle_id' and c.value = ? "
                 "join tags e on e.run_uuid = r.run_uuid and e.key = 'etapa' and e.value = 'etl' "
                 "where r.status = 'FINISHED' order by r.start_time desc limit 1")
            linha = con.execute(q, (cycle_id,)).fetchone()
            if not linha:
                return {}
            run = linha[0]
            met = dict(con.execute("select key, value from latest_metrics where run_uuid = ?", (run,)))
            par = dict(con.execute("select key, value from params where run_uuid = ?", (run,)))
        return {"metricas": met, "params": par}

    def lote_por_endereco(self) -> dict[str, str]:
        """address normalizado → lote, a partir dos lotes registrados pela ingestão."""
        id_col = self.cfg["dados"]["id"]
        mapa: dict[str, str] = {}
        for lote in ler_registro(self.cfg).get("lotes", []):
            arq = self._na_raiz(lote["arquivo"])
            if not arq.exists():
                continue
            for a in pd.read_csv(arq, usecols=[id_col])[id_col].astype(str):
                mapa.setdefault(a.strip().lower(), lote["nome"])   # 1ª ocorrência vence, como na ingestão
        return mapa


# --------------------------------------------------------------------------- #
# Montagem por ciclo
# --------------------------------------------------------------------------- #
def _ciclo(f: Fontes, cid: str, dec: dict, vig_antes: dict | None) -> dict:
    etl = f.run_etl(cid)
    met, par = etl.get("metricas", {}), etl.get("params", {})
    if not met:
        raise ValueError(f"{cid}: run 'etl' não encontrado no MLflow (necessário para tamanhos das partições)")
    resumo = f.resumo_ciclo(cid) or {}
    eleitas = (f.artefato_json(cid, "features_eleitas.json") or {}).get("features", [])
    d = dec["desafiante"]
    ant = dec.get("vigente_anterior")
    return {
        "cycle_id": cid,
        "executado_em": _iso_z(dec["decidido_em"]),
        "gatilho": resumo.get("gatilho") if resumo.get("gatilho") in ("schedule", "workflow_dispatch") else (
            "local" if cid.endswith("-local") else None),
        "execucao_url": resumo.get("github_run_url"),
        "data_version": dec["data_version"],
        "lotes": _lista(par.get("lotes")),
        "lotes_novos": _lista(par.get("lotes_novos")),
        "dados": {
            "n_carteiras": int(met["linhas"]),
            "duplicados_removidos": int(met.get("duplicados_removidos", 0)),
            "n_treino": int(met["treino_linhas"]),
            "n_validacao": int(met["validacao_linhas"]),
            "n_teste": int(met["teste_linhas"]),
            "taxa_fraude_treino": _r4(met["treino_taxa_fraude"]),
        },
        "n_features_eleitas": len(eleitas),
        "vencedor": {"modelo": d["modelo"], "algoritmo": d["algoritmo"], "run_id": d["run_id"],
                     "f1_val": _r4(d["f1_validacao"])},
        "vigente_anterior": None if not ant else {
            **_ref(ant.get("treinado_no_ciclo") or (vig_antes or {}).get("cycle_id"),
                   ant.get("modelo"), ant.get("algoritmo"), ant["run_id"]),
            "f1_val": _r4(ant["f1_validacao"]),
        },
        "qualidade": None if not dec.get("qualidade") else {
            "aprovada": bool(dec["qualidade"]["aprovada"]),
            "falhas": list(dec["qualidade"].get("falhas") or []),
        },
        "decisao": dec["decisao"],
        "motivo": dec["motivo"],
        "vigente_apos": None if not dec.get("run_id") else _ref(
            dec["treinado_no_ciclo"], dec["modelo"], dec["algoritmo"], dec["run_id"]),
    }


def _votacao(f: Fontes, cid: str, dec: dict, ciclo: dict, metodos: list[str]) -> dict:
    tab = f.artefato_csv(cid, "tabela_votos.csv")
    fe = f.artefato_json(cid, "features_eleitas.json")
    res = f.artefato_json(cid, "resumo_votacao.json")
    if tab is None or fe is None or res is None:
        raise ValueError(f"{cid}: faltam artefatos da votação em artifacts/{cid}/")
    eleita = tab["eleita"].astype(str).str.lower() == "true"
    linhas = []
    for (_, l), e in zip(tab.iterrows(), eleita):
        motivo = l.get("motivo")
        linhas.append({
            "feature": l["feature"],
            "votos": {m: int(l[f"voto_{m}"]) for m in metodos},
            "ranks": {m: int(l[f"rank_{m}"]) for m in metodos},
            "total_votos": int(l["total_votos"]),
            "rank_medio": round(float(l["rank_medio"]), 3),
            "eleita": bool(e),
            "motivo": None if (motivo is None or (isinstance(motivo, float) and math.isnan(motivo))) else motivo,
        })
    chaves = ("n_features_avaliadas", "n_eleitas", "n_eleitas_por_votos", "n_completadas_pelo_minimo",
              "n_com_3_votos", "n_com_2_votos", "n_com_1_voto", "n_sem_votos")
    return {
        "cycle_id": cid,
        "data_version": dec["data_version"],
        "mlflow_run_id": fe.get("mlflow_run_id"),
        "n_treino": ciclo["dados"]["n_treino"],
        "resumo": {k: int(res[k]) for k in chaves},
        "eleitas": [l["feature"] for l in linhas if l["eleita"]],
        "tabela": linhas,
    }


def _competicao(f: Fontes, cid: str, dec: dict, features_por_run: dict[str, list[str]]) -> dict:
    comp = f.artefato_json(cid, "competicao.json")
    if comp is None:
        raise ValueError(f"{cid}: falta artifacts/{cid}/competicao.json")
    modelos = []
    for papel in ("A", "B"):
        c = comp["candidatos"][papel]
        modelos.append({
            "papel": papel, "algoritmo": c["algoritmo"], "run_id": c["run_id"], "cycle_id_treino": cid,
            "features": list(comp["features"]),
            "metricas": _metricas(c["metricas_validacao"]),
            "matriz_confusao": _matriz(c["metricas_validacao"]),
        })
    ant = dec.get("vigente_anterior")
    if ant:
        mv = ant.get("metricas_validacao") or {}
        modelos.append({
            "papel": "vigente", "algoritmo": ant["algoritmo"], "run_id": ant["run_id"],
            "cycle_id_treino": ant["treinado_no_ciclo"],
            "features": features_por_run.get(ant["run_id"], []),
            "metricas": _metricas(mv),
            "matriz_confusao": _matriz(mv),
        })
    n_val = int(comp["candidatos"]["A"]["metricas_validacao"]["n"])
    return {"cycle_id": cid, "data_version": dec["data_version"], "n_validacao": n_val,
            "vencedor": dec["desafiante"]["modelo"], "decisao": dec["decisao"], "modelos": modelos}


def _tipo(prev: int, real: int) -> str:
    return {(1, 1): "VP", (0, 0): "VN", (1, 0): "FP", (0, 1): "FN"}[(prev, real)]


def _previsoes(f: Fontes, cid: str, dec: dict, ciclo: dict, lotes: dict[str, str]) -> dict | None:
    df = f.artefato_csv(cid, "previsoes.csv")
    if df is None or ciclo["vigente_apos"] is None:
        return None
    resumo = f.artefato_json(cid, "previsoes_resumo.json") or {}
    limiar = float(resumo.get("limiar", 0.5))
    p, y = df["previsao"].astype(int), df["rotulo_real"].astype(int)
    vp, vn = int(((p == 1) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    fp, fn = int(((p == 1) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
    prec = vp / (vp + fp) if vp + fp else 0.0
    rec = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    df = df.assign(_tipo=[_tipo(a, b) for a, b in zip(p, y)])
    partes = [g.sample(min(AMOSTRA_POR_TIPO[t], len(g)), random_state=SEED_AMOSTRA)
              for t, g in df.groupby("_tipo") if t in AMOSTRA_POR_TIPO]
    amostra = (pd.concat(partes) if partes else df.head(0)).sort_values("probabilidade_fraude", ascending=False)
    linhas = []
    for _, l in amostra.iterrows():
        addr = str(l["address"]).strip().lower()
        if addr not in lotes:
            raise ValueError(f"{cid}: address {addr} da amostra não está em nenhum lote registrado")
        linhas.append({"address": addr, "lote": lotes[addr],
                       "probabilidade": float(l["probabilidade_fraude"]),
                       "previsao": int(l["previsao"]), "rotulo_real": int(l["rotulo_real"])})
    criterio = ", ".join(f"{n} {t}" for t, n in AMOSTRA_POR_TIPO.items())
    return {
        "cycle_id": cid, "data_version": dec["data_version"], "modelo": ciclo["vigente_apos"],
        "particao": "teste", "limiar": limiar, "n_total": int(len(df)),
        "metricas": {"precision": _r4(prec), "recall": _r4(rec), "f1": _r4(f1)},
        "matriz_confusao": {"vn": vn, "fp": fp, "fn": fn, "vp": vp},
        "criterio_amostra": f"até {criterio} sorteados com seed {SEED_AMOSTRA}; ordenada pela probabilidade",
        "amostra": linhas,
    }


def _vigente(man: dict | None, ciclos: list[dict], regra: dict, env: dict) -> dict:
    vig = None
    if man and man.get("run_id"):
        vig = {
            **_ref(man["treinado_no_ciclo"], man["modelo"], man["algoritmo"], man["run_id"]),
            "data_version": man["treinado_com_data_version"],
            "features": list(man["features"]),
            "f1_val": _r4(man["f1_validacao"]),
            "f1_teste": None if man.get("f1_teste") is None else _r4(man["f1_teste"]),
            "promovido_em": _iso_z(man["promovido_em"]),
        }
    ult = ciclos[-1] if ciclos else None
    return {
        **env,
        "regra_promocao": {"metrica": "f1", "margem_minima": float(regra["margem_minima"]),
                           "f1_minimo": float(regra["f1_minimo"])},
        "vigente": vig,
        "ultimo_ciclo": None if not ult else {
            "cycle_id": ult["cycle_id"], "executado_em": ult["executado_em"],
            "data_version": ult["data_version"], "decisao": ult["decisao"], "motivo": ult["motivo"],
            "vencedor": ult["vencedor"],
            "f1_vigente_anterior": (ult["vigente_anterior"] or {}).get("f1_val"),
        },
        "modelo_mudou": bool(ult and ult["decisao"] == "promovido"),
        "anterior": ult["vigente_anterior"] if ult and ult["decisao"] == "promovido" else None,
    }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
def montar(fontes: Fontes | None = None) -> dict[str, dict]:
    """Os cinco JSONs do painel, em memória."""
    f = fontes or Fontes()
    cfg = f.cfg
    env = {"versao_formato": VERSAO_FORMATO, "gerado_em": _agora_z(), "origem": "pipeline"}
    vcfg = cfg.get("votacao", {})
    metodos = list(vcfg.get("metodos", []))
    regra_vot = {"top_k": int(vcfg["top_k"]), "votos_minimos": int(vcfg["votos_minimos"]),
                 "desempate": str(vcfg.get("desempate", "rank_medio")),
                 "minimo_features": int(vcfg.get("minimo_features", 5))}
    pcfg = cfg.get("promocao", {})

    hist = f.historico()
    ids = list(dict.fromkeys(hist["cycle_id"])) if not hist.empty else []
    lotes = f.lote_por_endereco() if ids else {}

    ciclos, votos, comps, prevs = [], [], [], []
    features_por_run: dict[str, list[str]] = {}
    vig_corrente: dict | None = None
    regra_prom = {"margem_minima": pcfg.get("margem_minima", 0.01), "f1_minimo": pcfg.get("f1_minimo", 0.6)}
    for cid in ids:
        dec = f.artefato_json(cid, "decisao.json")
        if dec is None:
            raise ValueError(f"{cid}: está no histórico, mas falta artifacts/{cid}/decisao.json")
        regra_prom = dec.get("regra") or regra_prom
        c = _ciclo(f, cid, dec, vig_corrente)
        ciclos.append(c)
        votos.append(_votacao(f, cid, dec, c, metodos))
        comps.append(_competicao(f, cid, dec, features_por_run))
        p = _previsoes(f, cid, dec, c, lotes)
        if p:
            prevs.append(p)
        if dec.get("run_id"):
            features_por_run[dec["run_id"]] = list(dec.get("features") or [])
        vig_corrente = c["vigente_apos"]

    return {
        "ciclos": {**env, "ciclos": ciclos},
        "votacao": {**env, "regra": regra_vot,
                    "metodos": [{"id": m, "nome": NOME_METODO.get(m, m)} for m in metodos],
                    "ciclos": votos},
        "competicao": {**env, "metricas": METRICAS, "ciclos": comps},
        "vigente": _vigente(f.manifesto(), ciclos, regra_prom, env),
        "previsoes": {**env, "ciclos": prevs},
    }


def exportar(saida: str | Path = SAIDA_PADRAO, fontes: Fontes | None = None) -> Path:
    """Monta, valida e grava. Se a validação falhar, não grava nada."""
    dados = montar(fontes)
    erros = validar_formato(dados) or validar_coerencia(dados)
    if erros:
        raise ValueError("JSONs do painel não passam no contrato:\n  - " + "\n  - ".join(erros))
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    for nome in ARQUIVOS:
        texto = json.dumps(dados[nome], ensure_ascii=False, indent=2) + "\n"
        tmp = saida / f"{nome}.json.tmp"
        tmp.write_text(texto, encoding="utf-8")
        tmp.replace(saida / f"{nome}.json")
    return saida


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--saida", default=str(SAIDA_PADRAO), help="pasta de saída (padrão: public/data)")
    a = ap.parse_args()
    try:
        pasta = exportar(a.saida)
    except (ValueError, FileNotFoundError, KeyError) as e:
        print(f"ERRO ao exportar o painel: {e}", file=sys.stderr)
        sys.exit(1)
    n = len(json.loads((pasta / "ciclos.json").read_text(encoding="utf-8"))["ciclos"])
    print(f"painel exportado em {pasta} ({n} ciclo(s))")


if __name__ == "__main__":
    _cli()
