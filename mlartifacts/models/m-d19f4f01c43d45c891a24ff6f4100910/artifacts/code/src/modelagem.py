"""Competição de modelos (Dev 3).

Cada candidato é um ``Pipeline`` do scikit-learn com três passos:

    pre      -- pré-processador do Dev 1 (log com sinal -> imputer -> scaler),
                sobre TODAS as features candidatas do ciclo;
    selecao  -- seleciona por nome as features eleitas pelo Dev 2;
    modelo   -- o estimador (A = Regressão Logística, B = XGBoost).

Tudo é ajustado com um único ``fit`` no treino, então o pré-processamento e a
seleção nunca enxergam validação ou teste. O modelo serializado no MLflow já
carrega o pré-processamento e a lista de features: basta passar as colunas
originais das partições (``data/processed/*.parquet``) para prever.

Os dois modelos usam as MESMAS features eleitas, o MESMO treino e são
avaliados na MESMA validação. O teste não é usado aqui.

Uso:
    from src.modelagem import resolver_ciclo, treinar_candidatos, registrar_candidato
    ciclo = resolver_ciclo()
    candidatos = treinar_candidatos(treino, validacao, ciclo["features"], ciclo["eleitas"])
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

from src.features import carregar_features_eleitas, configurar_mlflow, dir_artifacts
from src.ingestao import RAIZ, carregar_config
from src.preprocessamento import ALVO, carregar_schema, montar_preprocessador

ETAPA = "treino_competicao"
ARTEFATO_MODELO = "modelo"              # runs:/<run_id>/modelo
ARQ_COMPETICAO = "competicao.json"
ARQ_COMPARACAO = "comparacao_modelos.csv"
METRICAS = ("f1", "precision", "recall", "roc_auc", "average_precision")


# --------------------------------------------------------------------------- #
# Ciclo: cycle_id, data_version e features eleitas
# --------------------------------------------------------------------------- #
def resolver_ciclo(cycle_id: str | None = None, cfg: dict | None = None) -> dict:
    """Identifica o ciclo a partir do ``features_eleitas.json`` do Dev 2.

    - ``cycle_id`` explícito ou variável ``CYCLE_ID`` (workflow): usa exatamente
      esse ciclo e falha se a votação dele não existir;
    - sem nenhum dos dois (execução local): usa a votação mais recente e adota
      o ``cycle_id`` dela, para que 02 -> 03 -> 04 fiquem no mesmo ciclo mesmo
      quando o fallback ``<data-hora>-local`` muda de minuto entre notebooks.

    Também confere que a votação foi feita sobre o ``data_version`` atual das
    partições, para nunca treinar com features eleitas em dados antigos.
    """
    cfg = cfg or carregar_config()
    cid = cycle_id or os.environ.get("CYCLE_ID", "").strip() or None
    eleitas = carregar_features_eleitas(cid, cfg)
    schema = carregar_schema(cfg)
    if eleitas["data_version"] != schema["data_version"]:
        raise ValueError(
            f"features_eleitas.json do ciclo {eleitas['cycle_id']} é do data_version "
            f"{eleitas['data_version']}, mas as partições estão em {schema['data_version']}. "
            "Rode o notebook 02 de novo antes do 03."
        )
    faltando = [f for f in eleitas["features"] if f not in schema["features"]]
    if faltando:
        raise ValueError(f"features eleitas ausentes do schema atual: {faltando}")
    return {
        "cycle_id": eleitas["cycle_id"],
        "data_version": schema["data_version"],
        "features": list(schema["features"]),       # candidatas (entrada do pré-processador)
        "eleitas": list(eleitas["features"]),       # usadas pelos dois modelos
        "votacao_run_id": eleitas.get("mlflow_run_id"),
    }


# --------------------------------------------------------------------------- #
# Montagem dos candidatos
# --------------------------------------------------------------------------- #
def montar_selecao(eleitas: list[str]) -> ColumnTransformer:
    """Seleciona as features eleitas por nome (saída pandas do pré-processador)."""
    ct = ColumnTransformer([("eleitas", "passthrough", list(eleitas))],
                           remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def _hiperparametros(cfg: dict, algoritmo: str) -> dict:
    return dict(cfg.get("modelos", {}).get("hiperparametros", {}).get(algoritmo, {}))


def razao_classes(y: pd.Series) -> float:
    """legítimas / fraudes no treino: ``scale_pos_weight`` do XGBoost."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        raise ValueError("treino sem nenhuma fraude: não dá para treinar")
    return round(neg / pos, 6)


def montar_estimador(algoritmo: str, y_treino: pd.Series, cfg: dict | None = None):
    """Devolve ``(estimador, params)`` com seed fixa e balanceamento de classes."""
    cfg = cfg or carregar_config()
    seed = int(cfg["seeds"]["global"])
    hp = _hiperparametros(cfg, algoritmo)
    if algoritmo == "logistic_regression":
        params = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced", **hp,
                  "random_state": seed}
        return LogisticRegression(**params), params
    if algoritmo == "xgboost":
        from xgboost import XGBClassifier

        params = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                  "subsample": 0.9, "colsample_bytree": 0.9, **hp,
                  "scale_pos_weight": razao_classes(y_treino),
                  "eval_metric": "logloss", "tree_method": "hist",
                  "random_state": seed, "n_jobs": 1}   # 1 thread: resultado idêntico entre máquinas
        return XGBClassifier(**params), params
    raise ValueError(f"algoritmo desconhecido: {algoritmo}")


def montar_pipeline(features: list[str], eleitas: list[str], estimador,
                    cfg: dict | None = None) -> Pipeline:
    """pré-processamento (Dev 1) -> seleção das eleitas (Dev 2) -> estimador."""
    return Pipeline([
        ("pre", montar_preprocessador(features, cfg)),
        ("selecao", montar_selecao(eleitas)),
        ("modelo", estimador),
    ])


# --------------------------------------------------------------------------- #
# Avaliação
# --------------------------------------------------------------------------- #
def limiar(cfg: dict | None = None) -> float:
    cfg = cfg or carregar_config()
    return float(cfg.get("modelos", {}).get("limiar_classificacao", 0.5))


def avaliar(modelo, df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Precision, recall e F1 da classe fraude + matriz de confusão.

    ``df`` é uma partição como sai do parquet (``flag`` + features); o pipeline
    escolhe sozinho as colunas de que precisa.
    """
    y = df[ALVO].astype(int).to_numpy()
    proba = modelo.predict_proba(df)[:, 1]
    pred = (proba >= limiar(cfg)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
        "average_precision": float(average_precision_score(y, proba)) if y.any() else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n": int(len(y)),
    }


# --------------------------------------------------------------------------- #
# Treino dos dois candidatos
# --------------------------------------------------------------------------- #
def candidatos_configurados(cfg: dict | None = None) -> dict[str, str]:
    """``{"A": "logistic_regression", "B": "xgboost"}`` lido do config."""
    cfg = cfg or carregar_config()
    m = cfg["modelos"]
    return {"A": str(m["A"]), "B": str(m["B"])}


def treinar_candidato(nome: str, algoritmo: str, treino: pd.DataFrame,
                      features: list[str], eleitas: list[str], cfg: dict | None = None) -> dict:
    cfg = cfg or carregar_config()
    y = treino[ALVO].astype(int)
    estimador, params = montar_estimador(algoritmo, y, cfg)
    pipe = montar_pipeline(features, eleitas, estimador, cfg).fit(treino, y)
    return {"modelo": nome, "algoritmo": algoritmo, "params": params, "pipeline": pipe}


def treinar_candidatos(treino: pd.DataFrame, validacao: pd.DataFrame, features: list[str],
                       eleitas: list[str], cfg: dict | None = None) -> dict[str, dict]:
    """Treina A e B no mesmo treino, com as mesmas eleitas, e avalia na validação.

    Devolve ``{"A": {...}, "B": {...}}``, cada um com ``pipeline``, ``params``
    e ``metricas_validacao``.
    """
    cfg = cfg or carregar_config()
    saida = {}
    for nome, algoritmo in candidatos_configurados(cfg).items():
        c = treinar_candidato(nome, algoritmo, treino, features, eleitas, cfg)
        c["metricas_validacao"] = avaliar(c["pipeline"], validacao, cfg)
        saida[nome] = c
    return saida


def tabela_comparativa(candidatos: dict[str, dict]) -> pd.DataFrame:
    linhas = []
    for nome, c in candidatos.items():
        m = c["metricas_validacao"]
        linhas.append({"modelo": nome, "algoritmo": c["algoritmo"],
                       **{k: round(m[k], 4) for k in METRICAS},
                       "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]})
    return pd.DataFrame(linhas).set_index("modelo")


# --------------------------------------------------------------------------- #
# MLflow: um run por modelo
# --------------------------------------------------------------------------- #
def _assinatura(pipe: Pipeline, exemplo: pd.DataFrame):
    from mlflow.models import infer_signature

    return infer_signature(exemplo, pipe.predict_proba(exemplo)[:, 1])


def registrar_candidato(candidato: dict, ciclo: dict, validacao: pd.DataFrame,
                        cfg: dict | None = None, figuras: dict | None = None) -> str:
    """Abre um run ``treino_competicao`` e loga params, métricas, features e o modelo.

    Tags: ``cycle_id``, ``data_version``, ``algoritmo``, ``modelo`` (A/B),
    ``etapa`` e ``votacao_run_id``. Devolve o ``run_id``.
    """
    cfg = cfg or carregar_config()
    mlflow = configurar_mlflow(cfg)
    import mlflow.sklearn  # noqa: F401  (registra o flavor)

    m = candidato["metricas_validacao"]
    nome, algoritmo = candidato["modelo"], candidato["algoritmo"]
    with mlflow.start_run(run_name=f"{ETAPA}_{nome}_{algoritmo}_{ciclo['cycle_id']}") as run:
        mlflow.set_tags({
            "cycle_id": ciclo["cycle_id"], "data_version": ciclo["data_version"],
            "etapa": ETAPA, "modelo": nome, "algoritmo": algoritmo,
            "votacao_run_id": ciclo.get("votacao_run_id") or "",
        })
        mlflow.log_params({**{f"hp.{k}": v for k, v in candidato["params"].items()},
                           "seed": cfg["seeds"]["global"],
                           "limiar_classificacao": limiar(cfg),
                           "n_features": len(ciclo["eleitas"]),
                           "features": ",".join(ciclo["eleitas"])})
        mlflow.log_metrics({f"val_{k}": v for k, v in m.items() if not np.isnan(v)})
        mlflow.log_dict({"features": ciclo["eleitas"], "n_features": len(ciclo["eleitas"])},
                        "features.json")
        mlflow.log_dict({"labels": ["legitima", "fraude"],
                         "matriz": [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]},
                        "matriz_confusao_validacao.json")
        for caminho, fig in (figuras or {}).items():
            mlflow.log_figure(fig, caminho)

        # float: colunas inteiras podem ter ausentes em dados novos (assinatura não deve travar)
        exemplo = validacao[ciclo["features"]].head(5).astype("float64")
        kwargs = dict(sk_model=candidato["pipeline"], signature=_assinatura(candidato["pipeline"], exemplo),
                      code_paths=[str(RAIZ / "src")],   # o pipeline usa funções de src/
                      # cloudpickle: aceita o FunctionTransformer(log_com_sinal) do Dev 1
                      # (o formato skops, padrão nas versões novas, recusa funções do projeto)
                      serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE)
        try:
            mlflow.sklearn.log_model(name=ARTEFATO_MODELO, **kwargs)          # MLflow >= 3
        except TypeError:
            mlflow.sklearn.log_model(artifact_path=ARTEFATO_MODELO, **kwargs)  # MLflow 2.x
    candidato["run_id"] = run.info.run_id
    return run.info.run_id


def model_uri(run_id: str) -> str:
    return f"runs:/{run_id}/{ARTEFATO_MODELO}"


def carregar_modelo(run_id: str, cfg: dict | None = None):
    """Carrega o pipeline (sklearn) logado no run ``run_id``."""
    mlflow = configurar_mlflow(cfg)
    import mlflow.sklearn

    return mlflow.sklearn.load_model(model_uri(run_id))


# --------------------------------------------------------------------------- #
# Artefato do ciclo (contrato 03 -> 04 e painel do Dev 5)
# --------------------------------------------------------------------------- #
def gravar_competicao(candidatos: dict[str, dict], ciclo: dict, cfg: dict | None = None,
                      dir_saida: Path | None = None) -> Path:
    """Grava ``artifacts/<cycle_id>/competicao.json`` e ``comparacao_modelos.csv``."""
    cfg = cfg or carregar_config()
    pasta = Path(dir_saida) if dir_saida else dir_artifacts(cfg) / ciclo["cycle_id"]
    pasta.mkdir(parents=True, exist_ok=True)
    conteudo = {
        "cycle_id": ciclo["cycle_id"],
        "data_version": ciclo["data_version"],
        "etapa": ETAPA,
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "votacao_run_id": ciclo.get("votacao_run_id"),
        "features": ciclo["eleitas"],
        "limiar_classificacao": limiar(cfg),
        "candidatos": {
            nome: {"modelo": nome, "algoritmo": c["algoritmo"], "run_id": c.get("run_id"),
                   "model_uri": model_uri(c["run_id"]) if c.get("run_id") else None,
                   "params": c["params"], "metricas_validacao": c["metricas_validacao"]}
            for nome, c in candidatos.items()
        },
    }
    arq = pasta / ARQ_COMPETICAO
    arq.write_text(json.dumps(conteudo, indent=2, ensure_ascii=False, default=float) + "\n",
                   encoding="utf-8")
    tabela_comparativa(candidatos).to_csv(pasta / ARQ_COMPARACAO)
    return arq


def carregar_competicao(cycle_id: str | None = None, cfg: dict | None = None) -> dict:
    """Lê ``artifacts/<cycle_id>/competicao.json`` (sem id: o mais recente)."""
    cfg = cfg or carregar_config()
    base = dir_artifacts(cfg)
    cid = cycle_id or os.environ.get("CYCLE_ID", "").strip()
    if cid:
        arquivos = [base / cid / ARQ_COMPETICAO]
    else:
        arquivos = sorted(base.glob(f"*/{ARQ_COMPETICAO}"), key=lambda p: p.stat().st_mtime,
                          reverse=True)
    if not arquivos or not arquivos[0].exists():
        raise FileNotFoundError(f"nenhum {ARQ_COMPETICAO} em {base}; rode o notebook 03 antes")
    return json.loads(arquivos[0].read_text(encoding="utf-8"))
