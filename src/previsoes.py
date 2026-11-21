"""Previsões com o modelo vigente e comparação entre ciclos (Dev 4, notebook 05).

- ``carregar_vigente``: lê ``models/modelo_vigente.json`` (Dev 3) e carrega o
  pipeline do MLflow pelo ``run_id``. O pipeline já contém o pré-processamento
  do Dev 1 e a seleção das features eleitas do Dev 2, então recebe as colunas
  da partição como estão no parquet.
- ``prever``: probabilidade de fraude, classe prevista (limiar do config) e
  rótulo real, por carteira.
- ``historico_ciclos``: o que mudou entre ciclos (data_version, features,
  modelo e decisão), a partir de ``historico_decisoes.csv``, ``lotes.json`` e
  ``artifacts/<cycle_id>/features_eleitas.json``.

Só a partição de TESTE é usada aqui: ela nunca participou de nenhuma escolha
(votação, competição ou promoção).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingestao import RAIZ, carregar_config, ler_registro
from src.preprocessamento import ALVO, ID

ETAPA = "previsoes"
ARQ_PREVISOES = "previsoes.csv"
ARQ_RESUMO = "previsoes_resumo.json"


def _caminho(cfg: dict, chave: str, padrao: str) -> Path:
    return RAIZ / cfg.get("promocao", {}).get(chave, padrao)


def carregar_manifesto(cfg: dict | None = None) -> dict | None:
    """Manifesto do vigente ou ``None`` se ainda não há modelo promovido."""
    cfg = cfg or carregar_config()
    p = _caminho(cfg, "manifesto", "models/modelo_vigente.json")
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return m if m.get("run_id") else None


def carregar_vigente(cfg: dict | None = None):
    """``(manifesto, pipeline)`` do vigente; ``(None, None)`` se não houver."""
    cfg = cfg or carregar_config()
    man = carregar_manifesto(cfg)
    if man is None:
        return None, None
    from src.rastreio import configurar_mlflow

    mlflow = configurar_mlflow(cfg)
    import mlflow.sklearn  # noqa: F401

    uri = man.get("model_uri") or f"runs:/{man['run_id']}/modelo"
    return man, mlflow.sklearn.load_model(uri)


def limiar(cfg: dict | None = None) -> float:
    cfg = cfg or carregar_config()
    return float(cfg.get("modelos", {}).get("limiar_classificacao", 0.5))


def prever(modelo, df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Uma linha por carteira: ``address``, ``probabilidade_fraude``, ``previsao``,
    ``rotulo_real`` e ``acerto`` (se houver rótulo)."""
    proba = modelo.predict_proba(df)[:, 1]
    saida = pd.DataFrame({
        "address": df[ID].to_numpy() if ID in df else range(len(df)),
        "probabilidade_fraude": proba.round(6),
        "previsao": (proba >= limiar(cfg)).astype(int),
    })
    if ALVO in df:
        saida["rotulo_real"] = df[ALVO].astype(int).to_numpy()
        saida["acerto"] = saida["previsao"] == saida["rotulo_real"]
    return saida


def resumir(prev: pd.DataFrame) -> dict:
    """Contagens e métricas da classe fraude sobre as previsões."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    r = {"n": int(len(prev)), "previstas_fraude": int(prev["previsao"].sum())}
    if "rotulo_real" in prev:
        y, p = prev["rotulo_real"], prev["previsao"]
        r |= {
            "fraudes_reais": int(y.sum()),
            "tp": int(((p == 1) & (y == 1)).sum()), "fp": int(((p == 1) & (y == 0)).sum()),
            "fn": int(((p == 0) & (y == 1)).sum()), "tn": int(((p == 0) & (y == 0)).sum()),
            "f1": float(f1_score(y, p, zero_division=0)),
            "precision": float(precision_score(y, p, zero_division=0)),
            "recall": float(recall_score(y, p, zero_division=0)),
        }
    return r


def amostra(prev: pd.DataFrame, n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Amostra estável para exibir: metade fraudes reais, metade legítimas."""
    if "rotulo_real" not in prev:
        return prev.sample(min(n, len(prev)), random_state=seed)
    partes = [g.sample(min(n // 2, len(g)), random_state=seed)
              for _, g in prev.groupby("rotulo_real")]
    return pd.concat(partes).sort_values("probabilidade_fraude", ascending=False)


# --------------------------------------------------------------------------- #
# O que mudou entre ciclos
# --------------------------------------------------------------------------- #
def _features_do_ciclo(cycle_id: str, cfg: dict) -> list[str] | None:
    base = RAIZ / cfg.get("votacao", {}).get("dir_artifacts", "artifacts")
    p = base / str(cycle_id) / "features_eleitas.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get("features")


def historico_ciclos(cfg: dict | None = None) -> pd.DataFrame:
    """Uma linha por ciclo decidido, com o que mudou em relação ao anterior."""
    cfg = cfg or carregar_config()
    p = _caminho(cfg, "historico", "models/historico_decisoes.csv")
    if not p.exists():
        return pd.DataFrame()
    h = pd.read_csv(p)
    if h.empty:
        return h
    lotes: dict[str, list[str]] = {}
    for l in ler_registro(cfg).get("lotes", []):
        lotes.setdefault(str(l.get("cycle_id")), []).append(l["nome"])
    h["lotes_novos"] = h["cycle_id"].map(lambda c: ",".join(lotes.get(str(c), [])))
    h["features_eleitas"] = h["cycle_id"].map(lambda c: _features_do_ciclo(c, cfg))
    h["n_features"] = h["features_eleitas"].map(lambda f: len(f) if f else None)

    anteriores = h.shift(1)
    h["mudou_data_version"] = h["data_version"] != anteriores["data_version"]
    h["mudou_features"] = [
        (a is None or b is None or set(a) != set(b)) if i > 0 else True
        for i, (a, b) in enumerate(zip(h["features_eleitas"], anteriores["features_eleitas"]))
    ]
    h["mudou_modelo"] = h["vigente_run_id"] != anteriores["vigente_run_id"]
    h.loc[h.index[0], ["mudou_data_version", "mudou_modelo"]] = True
    return h


def diff_features(anterior: list[str] | None, atual: list[str] | None) -> dict:
    a, b = set(anterior or []), set(atual or [])
    return {"entraram": sorted(b - a), "sairam": sorted(a - b), "mantidas": sorted(a & b)}


# --------------------------------------------------------------------------- #
def gravar_previsoes(prev: pd.DataFrame, resumo: dict, manifesto: dict, cycle_id: str,
                     data_version: str, cfg: dict | None = None) -> dict:
    """Grava ``artifacts/<cycle_id>/previsoes.csv`` e ``previsoes_resumo.json``."""
    cfg = cfg or carregar_config()
    pasta = RAIZ / cfg.get("votacao", {}).get("dir_artifacts", "artifacts") / cycle_id
    pasta.mkdir(parents=True, exist_ok=True)
    prev.to_csv(pasta / ARQ_PREVISOES, index=False, lineterminator="\n")
    conteudo = {
        "cycle_id": cycle_id, "data_version": data_version, "etapa": ETAPA,
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "particao": "teste", "limiar": limiar(cfg),
        "vigente": {k: manifesto.get(k) for k in ("run_id", "modelo", "algoritmo",
                                                   "treinado_no_ciclo", "promovido_no_ciclo",
                                                   "features")},
        "resumo": resumo,
    }
    (pasta / ARQ_RESUMO).write_text(json.dumps(conteudo, indent=2, ensure_ascii=False) + "\n",
                                    encoding="utf-8")
    return {"previsoes": pasta / ARQ_PREVISOES, "resumo": pasta / ARQ_RESUMO}
