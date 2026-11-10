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

Uso:
    from src.features import preparar_treino, executar_metodos
    X, y = preparar_treino(treino, features)
    resultados = executar_metodos(X, y)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from src.ingestao import carregar_config
from src.preprocessamento import ALVO, montar_preprocessador

METODOS = ("selectkbest_f_classif", "random_forest", "rfe_logistic")


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
    return (top_k if top_k is not None else int(cfg["votacao"]["top_k"]),
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
def executar_metodos(X: pd.DataFrame, y: pd.Series, cfg: dict | None = None) -> dict[str, dict]:
    """Roda os três métodos listados em ``config.yaml`` e devolve {nome: resultado}."""
    cfg = cfg or carregar_config()
    disponiveis = {
        "selectkbest_f_classif": lambda: selecionar_kbest(X, y, cfg=cfg),
        "random_forest": lambda: selecionar_random_forest(X, y, cfg=cfg),
        "rfe_logistic": lambda: selecionar_rfe(X, y, cfg=cfg),
    }
    resultados = {}
    for nome in cfg["votacao"]["metodos"]:
        if nome not in disponiveis:
            raise ValueError(f"método de seleção desconhecido no config.yaml: {nome}")
        resultados[nome] = disponiveis[nome]()
    return resultados
