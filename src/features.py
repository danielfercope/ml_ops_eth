"""Seleção de features por três métodos (Dev 2).

Cada método recebe o treino JÁ pré-processado (pré-processador do Dev 1
ajustado só no treino) e devolve o ranking completo das features mais o
top-k. Nenhum método olha para validação ou teste.

Métodos (nomes iguais aos de ``config.yaml`` -> ``votacao.metodos``):
    - ``selectkbest_f_classif``: SelectKBest com estatística F (ANOVA).
    - ``random_forest``: importância (Gini) de um RandomForest balanceado.
    - ``rfe_logistic``: RFE com Regressão Logística balanceada, eliminando
      uma feature por vez até sobrar uma, o que dá a ordem completa.

Formato devolvido por cada método (dict):
    {
      "metodo":  "random_forest",
      "params":  {...},                # hiperparâmetros usados
      "ranking": DataFrame[feature, score, rank],   # rank 1 = melhor
      "top_k":   ["feature_a", ...],   # as k melhores, em ordem
    }

Votação (``votar``): cada método vota nas suas top-k; eleita com >= 2 de 3
votos; desempate/ordem pela média dos ranks; mínimo de 5 features completado
pelo rank médio. Regra lida de ``config.yaml`` por ``carregar_regra_votacao``.

Uso:
    from src.features import preparar_treino, executar_metodos, votar
    X, y = preparar_treino(treino, features)
    resultados = executar_metodos(X, y)
    tabela_votos, eleitas = votar(resultados)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from src.ingestao import RAIZ, carregar_config
from src.preprocessamento import (ALVO, carregar_particoes, carregar_schema,
                                  montar_preprocessador)

METODOS = ("selectkbest_f_classif", "random_forest", "rfe_logistic")
DESEMPATES = ("rank_medio",)


# --------------------------------------------------------------------------- #
# Regra de votação (fixada em config.yaml ANTES de ver qualquer resultado)
# --------------------------------------------------------------------------- #
def carregar_regra_votacao(cfg: dict | None = None) -> dict:
    """Lê e valida a seção ``votacao`` do ``config.yaml``.

    A regra foi gravada antes do ciclo 1 e não deve mudar depois de ver
    resultados (critério 4: "regra objetiva de votação"). Esta função é o
    único ponto de leitura da regra, então notebook, votação e MLflow usam
    exatamente os mesmos valores.

    Campos:
        metodos        -- os 3 métodos, na ordem em que aparecem na tabela
        top_k          -- quantas features cada método "vota" (as k melhores)
        votos_minimos  -- votos necessários para ser eleita (2 de 3)
        desempate      -- 'rank_medio': média dos ranks nos 3 métodos
        minimo_features-- se menos features forem eleitas, completa-se com as
                          de melhor rank médio até atingir este mínimo
    """
    cfg = cfg or carregar_config()
    v = cfg.get("votacao")
    if not isinstance(v, dict):
        raise ValueError("config.yaml sem a seção 'votacao'")

    obrigatorios = ("metodos", "top_k", "votos_minimos", "desempate", "minimo_features")
    faltando = [c for c in obrigatorios if c not in v]
    if faltando:
        raise ValueError(f"seção 'votacao' sem os campos: {faltando}")

    metodos = list(v["metodos"])
    if len(metodos) != 3 or len(set(metodos)) != 3:
        raise ValueError("a votação exige exatamente 3 métodos distintos")
    desconhecidos = [m for m in metodos if m not in METODOS]
    if desconhecidos:
        raise ValueError(f"métodos desconhecidos: {desconhecidos}; válidos: {list(METODOS)}")

    top_k, votos, minimo = int(v["top_k"]), int(v["votos_minimos"]), int(v["minimo_features"])
    if top_k < 1:
        raise ValueError("top_k deve ser >= 1")
    if not 1 <= votos <= len(metodos):
        raise ValueError(f"votos_minimos deve estar entre 1 e {len(metodos)}")
    if not 1 <= minimo <= top_k:
        raise ValueError("minimo_features deve estar entre 1 e top_k")
    if v["desempate"] not in DESEMPATES:
        raise ValueError(f"desempate desconhecido: {v['desempate']}; válidos: {list(DESEMPATES)}")

    return {"metodos": metodos, "top_k": top_k, "votos_minimos": votos,
            "desempate": str(v["desempate"]), "minimo_features": minimo}


# --------------------------------------------------------------------------- #
# Preparação (pré-processador do Dev 1 ajustado SÓ no treino)
# --------------------------------------------------------------------------- #
def preparar_treino(treino: pd.DataFrame, features: list[str],
                    cfg: dict | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Ajusta o pré-processador no treino e devolve ``X`` (DataFrame) e ``y``."""
    cfg = cfg or carregar_config()
    pre = montar_preprocessador(features, cfg).fit(treino)
    X = pre.transform(treino)
    y = treino[ALVO].astype(int).reset_index(drop=True)
    return X.reset_index(drop=True), y


# --------------------------------------------------------------------------- #
# Utilitários
# --------------------------------------------------------------------------- #
def _montar_ranking(scores: pd.Series, maior_melhor: bool = True) -> pd.DataFrame:
    """Ordena por score (desempate pelo nome, para ser determinístico) e numera.

    ``rank`` vai de 1 (melhor) a n, sem empates: assim a média dos rankings
    usada no desempate da votação fica bem definida.
    """
    df = pd.DataFrame({"feature": scores.index, "score": scores.values})
    df["score"] = df["score"].astype(float).fillna(0.0)
    df = df.sort_values(["score", "feature"], ascending=[not maior_melhor, True],
                        kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def _resultado(metodo: str, params: dict, ranking: pd.DataFrame, top_k: int) -> dict:
    return {
        "metodo": metodo,
        "params": params,
        "ranking": ranking,
        "top_k": ranking.loc[ranking["rank"] <= top_k, "feature"].tolist(),
    }


def _top_k_e_seed(cfg: dict | None, top_k: int | None, seed: int | None) -> tuple[int, int]:
    cfg = cfg or carregar_config()
    return (top_k if top_k is not None else carregar_regra_votacao(cfg)["top_k"],
            seed if seed is not None else int(cfg["seeds"]["global"]))


# --------------------------------------------------------------------------- #
# Método 1: SelectKBest com f_classif
# --------------------------------------------------------------------------- #
def selecionar_kbest(X: pd.DataFrame, y: pd.Series, top_k: int | None = None,
                     cfg: dict | None = None) -> dict:
    """Estatística F (ANOVA) de cada feature contra o alvo. Maior F = melhor."""
    top_k, _ = _top_k_e_seed(cfg, top_k, None)
    seletor = SelectKBest(score_func=f_classif, k=min(top_k, X.shape[1])).fit(X, y)
    scores = pd.Series(seletor.scores_, index=X.columns)
    params = {"score_func": "f_classif", "k": int(seletor.k)}
    return _resultado("selectkbest_f_classif", params, _montar_ranking(scores), top_k)


# --------------------------------------------------------------------------- #
# Método 2: importância por Random Forest
# --------------------------------------------------------------------------- #
def selecionar_random_forest(X: pd.DataFrame, y: pd.Series, top_k: int | None = None,
                             seed: int | None = None, n_estimators: int = 300,
                             cfg: dict | None = None) -> dict:
    """Importância média de redução de impureza (Gini), floresta balanceada."""
    top_k, seed = _top_k_e_seed(cfg, top_k, seed)
    params = {"n_estimators": n_estimators, "class_weight": "balanced",
              "random_state": seed, "n_jobs": -1}
    rf = RandomForestClassifier(**params).fit(X, y)
    scores = pd.Series(rf.feature_importances_, index=X.columns)
    return _resultado("random_forest", params, _montar_ranking(scores), top_k)


# --------------------------------------------------------------------------- #
# Método 3: RFE com Regressão Logística
# --------------------------------------------------------------------------- #
def selecionar_rfe(X: pd.DataFrame, y: pd.Series, top_k: int | None = None,
                   seed: int | None = None, max_iter: int = 2000,
                   cfg: dict | None = None) -> dict:
    """Eliminação recursiva: remove 1 feature por vez até sobrar 1.

    Como o passo é 1 e a eliminação é determinística, as k que sobreviveriam
    num RFE até k são exatamente as de ``rank <= k`` aqui. Guardar a ordem
    completa permite calcular o rank médio na votação. ``score`` é a posição
    em que a feature foi eliminada (maior = sobreviveu mais tempo).
    """
    top_k, seed = _top_k_e_seed(cfg, top_k, seed)
    params_lr = {"class_weight": "balanced", "max_iter": max_iter, "random_state": seed}
    estimador = LogisticRegression(**params_lr)
    rfe = RFE(estimador, n_features_to_select=1, step=1).fit(X, y)
    # ranking_: 1 = última a sobrar ... n = primeira eliminada
    score = pd.Series(X.shape[1] - rfe.ranking_ + 1, index=X.columns)
    params = {"estimator": "LogisticRegression", **params_lr, "step": 1,
              "n_features_to_select_final": top_k}
    return _resultado("rfe_logistic", params, _montar_ranking(score), top_k)


# --------------------------------------------------------------------------- #
# Votação
# --------------------------------------------------------------------------- #
MOTIVO_VOTOS = "votos_suficientes"
MOTIVO_MINIMO = "completou_minimo"


def votar(resultados: dict[str, dict], regra: dict | None = None,
          cfg: dict | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Aplica a regra de votação aos resultados dos métodos.

    Regra (``config.yaml`` -> ``votacao``):
        1. cada método vota nas suas ``top_k`` features;
        2. é eleita quem tem ``votos_minimos`` ou mais votos;
        3. desempate / ordem: média dos ranks nos 3 métodos (menor = melhor);
        4. se sobrarem menos de ``minimo_features`` eleitas, completa-se com
           as não eleitas de melhor rank médio (motivo ``completou_minimo``).

    Devolve ``(tabela, eleitas)``:
        tabela  -- uma linha por feature: ``voto_<m>`` (0/1) e ``rank_<m>`` por
                   método, ``total_votos``, ``rank_medio``, ``eleita`` (bool) e
                   ``motivo``; ordenada por eleita, total de votos e rank médio.
        eleitas -- nomes das features eleitas, na ordem da tabela.
    """
    regra = regra or carregar_regra_votacao(cfg)
    metodos = regra["metodos"]
    faltando = [m for m in metodos if m not in resultados]
    if faltando:
        raise ValueError(f"faltam resultados dos métodos: {faltando}")

    base = resultados[metodos[0]]["ranking"]["feature"]
    for m in metodos[1:]:
        if set(resultados[m]["ranking"]["feature"]) != set(base):
            raise ValueError(f"o método {m} rankeou um conjunto diferente de features")

    tabela = pd.DataFrame({"feature": sorted(base)})
    for m in metodos:
        rk = resultados[m]["ranking"].set_index("feature")["rank"]
        tabela[f"rank_{m}"] = tabela["feature"].map(rk).astype(int)
        tabela[f"voto_{m}"] = tabela["feature"].isin(resultados[m]["top_k"]).astype(int)

    tabela["total_votos"] = tabela[[f"voto_{m}" for m in metodos]].sum(axis=1)
    tabela["rank_medio"] = tabela[[f"rank_{m}" for m in metodos]].mean(axis=1).round(3)

    tabela["eleita"] = tabela["total_votos"] >= regra["votos_minimos"]
    tabela["motivo"] = np.where(tabela["eleita"], MOTIVO_VOTOS, "")

    faltam = regra["minimo_features"] - int(tabela["eleita"].sum())
    if faltam > 0:
        candidatas = tabela.loc[~tabela["eleita"]].sort_values(
            ["rank_medio", "total_votos", "feature"], ascending=[True, False, True],
            kind="mergesort").head(faltam).index
        tabela.loc[candidatas, ["eleita", "motivo"]] = [True, MOTIVO_MINIMO]

    tabela = tabela.sort_values(
        ["eleita", "total_votos", "rank_medio", "feature"],
        ascending=[False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)

    colunas = (["feature"] + [f"voto_{m}" for m in metodos] + ["total_votos"]
               + [f"rank_{m}" for m in metodos] + ["rank_medio", "eleita", "motivo"])
    tabela = tabela[colunas]
    eleitas = tabela.loc[tabela["eleita"], "feature"].tolist()
    return tabela, eleitas


def resumir_votacao(tabela: pd.DataFrame, regra: dict) -> dict:
    """Números para log/relatório: quantas eleitas, por quantos votos, etc."""
    eleitas = tabela.loc[tabela["eleita"]]
    return {
        "n_features_avaliadas": int(len(tabela)),
        "n_eleitas": int(len(eleitas)),
        "n_eleitas_por_votos": int((eleitas["motivo"] == MOTIVO_VOTOS).sum()),
        "n_completadas_pelo_minimo": int((eleitas["motivo"] == MOTIVO_MINIMO).sum()),
        "n_com_3_votos": int((tabela["total_votos"] == 3).sum()),
        "n_com_2_votos": int((tabela["total_votos"] == 2).sum()),
        "n_com_1_voto": int((tabela["total_votos"] == 1).sum()),
        "n_sem_votos": int((tabela["total_votos"] == 0).sum()),
        "minimo_features": regra["minimo_features"],
        "votos_minimos": regra["votos_minimos"],
        "top_k": regra["top_k"],
    }


# --------------------------------------------------------------------------- #
def executar_metodos(X: pd.DataFrame, y: pd.Series, cfg: dict | None = None) -> dict[str, dict]:
    """Roda os três métodos listados em ``config.yaml`` e devolve {nome: resultado}."""
    cfg = cfg or carregar_config()
    regra = carregar_regra_votacao(cfg)
    disponiveis = {
        "selectkbest_f_classif": lambda: selecionar_kbest(X, y, cfg=cfg),
        "random_forest": lambda: selecionar_random_forest(X, y, cfg=cfg),
        "rfe_logistic": lambda: selecionar_rfe(X, y, cfg=cfg),
    }
    return {nome: disponiveis[nome]() for nome in regra["metodos"]}


# --------------------------------------------------------------------------- #
# Identificadores do ciclo e MLflow
# --------------------------------------------------------------------------- #
ETAPA = "votacao_features"
ARQ_ELEITAS = "features_eleitas.json"
ARQ_TABELA = "tabela_votos.csv"
ARQ_RESUMO = "resumo_votacao.json"


def obter_cycle_id() -> str:
    """``cycle_id`` do ciclo: do ``src.rastreio`` do Dev 4 se existir, senão
    da variável ``CYCLE_ID`` (definida pelo workflow), senão ``local-<data>``."""
    try:  # convenção do Dev 4, quando disponível
        from src import rastreio  # type: ignore
        if hasattr(rastreio, "obter_cycle_id"):
            return str(rastreio.obter_cycle_id())
    except ImportError:
        pass
    cid = os.environ.get("CYCLE_ID", "").strip()
    return cid or "local-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")


def dir_artifacts(cfg: dict | None = None) -> Path:
    cfg = cfg or carregar_config()
    return RAIZ / cfg.get("votacao", {}).get("dir_artifacts", "artifacts")


def tracking_uri(cfg: dict | None = None) -> str:
    """URI do MLflow: ``MLFLOW_TRACKING_URI`` se definida; senão a do config,
    com caminho SQLite relativo resolvido a partir da raiz do repositório
    (o mesmo ``mlflow.db`` seja o notebook rodado de ``notebooks/`` ou da raiz)."""
    env = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env:
        return env
    cfg = cfg or carregar_config()
    uri = str(cfg["projeto"]["tracking_uri"])
    prefixo = "sqlite:///"
    if uri.startswith(prefixo) and not Path(uri[len(prefixo):]).is_absolute():
        uri = prefixo + (RAIZ / uri[len(prefixo):]).resolve().as_posix()
    return uri


def configurar_mlflow(cfg: dict | None = None):
    """Aponta o MLflow para o tracking do projeto e garante o experimento."""
    import mlflow

    cfg = cfg or carregar_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    nome = cfg["projeto"]["experimento_mlflow"]
    if mlflow.get_experiment_by_name(nome) is None:
        mlflow.create_experiment(nome, artifact_location=(RAIZ / "mlartifacts").resolve().as_uri())
    mlflow.set_experiment(nome)
    return mlflow


def registrar_votacao(resultados: dict[str, dict], tabela: pd.DataFrame, eleitas: list[str],
                      regra: dict, data_version: str, cycle_id: str | None = None,
                      cfg: dict | None = None, dir_saida: Path | None = None) -> dict:
    """Grava os artefatos do ciclo em ``artifacts/<cycle_id>/`` e loga tudo no MLflow.

    MLflow (um run por ciclo, etapa ``votacao_features``):
        tags     cycle_id, data_version, etapa
        params   regra de votação + hiperparâmetros de cada método (prefixados)
        metrics  resumo da votação (nº eleitas, nº com 3/2/1/0 votos, ...)
        artifacts rankings/<metodo>.csv, tabela_votos.csv, features_eleitas.json

    Devolve dict com ``run_id``, ``cycle_id`` e os caminhos gravados.
    """
    cfg = cfg or carregar_config()
    cycle_id = cycle_id or obter_cycle_id()
    pasta = Path(dir_saida) if dir_saida else dir_artifacts(cfg) / cycle_id
    (pasta / "rankings").mkdir(parents=True, exist_ok=True)

    resumo = resumir_votacao(tabela, regra)
    mlflow = configurar_mlflow(cfg)
    with mlflow.start_run(run_name=f"{ETAPA}_{cycle_id}") as run:
        mlflow.set_tags({"cycle_id": cycle_id, "data_version": data_version, "etapa": ETAPA})

        mlflow.log_params({f"regra.{k}": (",".join(v) if isinstance(v, list) else v)
                           for k, v in regra.items()})
        for nome, r in resultados.items():
            mlflow.log_params({f"{nome}.{k}": v for k, v in r["params"].items()})
            r["ranking"].to_csv(pasta / "rankings" / f"{nome}.csv", index=False)

        mlflow.log_metrics({k: v for k, v in resumo.items() if isinstance(v, (int, float))})

        tabela.to_csv(pasta / ARQ_TABELA, index=False)
        eleitas_json = {
            "cycle_id": cycle_id,
            "data_version": data_version,
            "etapa": ETAPA,
            "mlflow_run_id": run.info.run_id,
            "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "regra": regra,
            "n_features_avaliadas": resumo["n_features_avaliadas"],
            "n_eleitas": len(eleitas),
            "features": eleitas,
            "motivo": dict(zip(tabela.loc[tabela["eleita"], "feature"],
                               tabela.loc[tabela["eleita"], "motivo"])),
            "top_k_por_metodo": {m: r["top_k"] for m, r in resultados.items()},
        }
        (pasta / ARQ_ELEITAS).write_text(json.dumps(eleitas_json, indent=2, ensure_ascii=False) + "\n",
                                         encoding="utf-8")
        (pasta / ARQ_RESUMO).write_text(json.dumps(resumo, indent=2, ensure_ascii=False) + "\n",
                                        encoding="utf-8")
        mlflow.log_artifacts(str(pasta))
        mlflow.log_param("features_eleitas", ",".join(eleitas))

    return {"run_id": run.info.run_id, "cycle_id": cycle_id, "data_version": data_version,
            "dir": pasta, "features_eleitas": pasta / ARQ_ELEITAS,
            "tabela_votos": pasta / ARQ_TABELA, "eleitas": eleitas}


def executar_votacao(cfg: dict | None = None, cycle_id: str | None = None) -> dict:
    """Fluxo completo do notebook 02: partições -> 3 métodos -> votação -> MLflow.

    Usa só o treino. Devolve ``resultados``, ``tabela``, ``eleitas``, ``regra``
    e o ``registro`` (run_id e caminhos).
    """
    cfg = cfg or carregar_config()
    regra = carregar_regra_votacao(cfg)
    schema = carregar_schema(cfg)
    treino, _, _ = carregar_particoes(cfg)
    X, y = preparar_treino(treino, schema["features"], cfg)
    resultados = executar_metodos(X, y, cfg)
    tabela, eleitas = votar(resultados, regra)
    registro = registrar_votacao(resultados, tabela, eleitas, regra,
                                 schema["data_version"], cycle_id, cfg)
    return {"resultados": resultados, "tabela": tabela, "eleitas": eleitas,
            "regra": regra, "registro": registro, "n_treino": int(len(X))}


def carregar_features_eleitas(cycle_id: str | None = None, cfg: dict | None = None) -> dict:
    """Lê ``artifacts/<cycle_id>/features_eleitas.json`` (contrato com o Dev 3).

    Sem ``cycle_id`` usa a variável ``CYCLE_ID`` e, na falta dela, o arquivo
    gerado mais recentemente.
    """
    cfg = cfg or carregar_config()
    base = dir_artifacts(cfg)
    cid = cycle_id or os.environ.get("CYCLE_ID", "").strip()
    if cid:
        arquivos = [base / cid / ARQ_ELEITAS]
    else:
        arquivos = sorted(base.glob(f"*/{ARQ_ELEITAS}"), key=lambda p: p.stat().st_mtime,
                          reverse=True)
    if not arquivos or not arquivos[0].exists():
        raise FileNotFoundError(f"nenhum {ARQ_ELEITAS} em {base}; rode o notebook 02 antes")
    return json.loads(arquivos[0].read_text(encoding="utf-8"))
