"""Testes dos três métodos de seleção de features (Dev 2).

1. Cada método devolve um ranking com TODAS as features, ranks 1..n sem
   empates e um top-k com exatamente k nomes (em ordem de rank).
2. Os três métodos são determinísticos (mesma entrada -> mesmo resultado).
3. Em dados sintéticos, as features informativas dominam o top-k.
4. Com os parquets do Dev 1 presentes, o fluxo completo roda sobre o treino.

Rodar a partir da raiz:  python -m pytest -q tests/test_features.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.features import (METODOS, executar_metodos, preparar_treino,
                          selecionar_kbest, selecionar_random_forest,
                          selecionar_rfe)
from src.ingestao import RAIZ, carregar_config

N_FEATURES, N_INFORMATIVAS, TOP_K = 12, 4, 5


@pytest.fixture(scope="module")
def dados_sinteticos():
    X, y = make_classification(n_samples=600, n_features=N_FEATURES,
                               n_informative=N_INFORMATIVAS, n_redundant=0,
                               n_repeated=0, n_clusters_per_class=1,
                               weights=[0.8, 0.2], shuffle=False, random_state=42)
    # shuffle=False: as N_INFORMATIVAS primeiras colunas são as informativas
    cols = [f"info_{i}" for i in range(N_INFORMATIVAS)] + \
           [f"ruido_{i}" for i in range(N_FEATURES - N_INFORMATIVAS)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="flag")


@pytest.fixture(scope="module")
def cfg_teste():
    cfg = carregar_config()
    cfg["votacao"]["top_k"] = TOP_K
    return cfg


METODO_FUNCS = {
    "selectkbest_f_classif": selecionar_kbest,
    "random_forest": selecionar_random_forest,
    "rfe_logistic": selecionar_rfe,
}


@pytest.mark.parametrize("nome", METODOS)
def test_formato_do_ranking_e_top_k(dados_sinteticos, cfg_teste, nome):
    X, y = dados_sinteticos
    r = METODO_FUNCS[nome](X, y, cfg=cfg_teste)

    assert r["metodo"] == nome
    assert isinstance(r["params"], dict) and r["params"]
    rk = r["ranking"]
    assert list(rk.columns) == ["feature", "score", "rank"]
    assert set(rk["feature"]) == set(X.columns)
    assert sorted(rk["rank"]) == list(range(1, N_FEATURES + 1))
    assert not rk["score"].isna().any()
    assert r["top_k"] == rk.sort_values("rank")["feature"].head(TOP_K).tolist()
    assert len(r["top_k"]) == TOP_K


@pytest.mark.parametrize("nome", METODOS)
def test_deterministico(dados_sinteticos, cfg_teste, nome):
    X, y = dados_sinteticos
    a = METODO_FUNCS[nome](X, y, cfg=cfg_teste)
    b = METODO_FUNCS[nome](X, y, cfg=cfg_teste)
    pd.testing.assert_frame_equal(a["ranking"], b["ranking"])
    assert a["top_k"] == b["top_k"]


@pytest.mark.parametrize("nome", METODOS)
def test_informativas_dominam_o_top_k(dados_sinteticos, cfg_teste, nome):
    X, y = dados_sinteticos
    r = METODO_FUNCS[nome](X, y, cfg=cfg_teste)
    informativas = {c for c in r["top_k"] if c.startswith("info_")}
    # top-5 deve conter pelo menos 3 das 4 informativas
    assert len(informativas) >= 3, r["top_k"]


def test_executar_metodos_segue_o_config(dados_sinteticos, cfg_teste):
    X, y = dados_sinteticos
    res = executar_metodos(X, y, cfg=cfg_teste)
    assert list(res) == cfg_teste["votacao"]["metodos"]
    assert set(res) == set(METODOS)


def test_metodo_desconhecido_da_erro(dados_sinteticos, cfg_teste):
    X, y = dados_sinteticos
    cfg = {**cfg_teste, "votacao": {**cfg_teste["votacao"],
                                    "metodos": ["pca", "random_forest", "rfe_logistic"]}}
    with pytest.raises(ValueError, match="desconhecido"):
        executar_metodos(X, y, cfg=cfg)


# --------------------------------------------------------------------------- #
# Integração com os parquets do Dev 1 (pula se o ETL ainda não rodou)
# --------------------------------------------------------------------------- #
PROCESSED = RAIZ / "data/processed"


@pytest.mark.skipif(not (PROCESSED / "treino.parquet").exists(),
                    reason="rode `python -m src.preprocessamento` antes")
def test_fluxo_no_treino_real():
    import json

    from src.preprocessamento import carregar_particoes
    schema = json.loads((PROCESSED / "schema.json").read_text("utf-8"))
    treino, _, _ = carregar_particoes()
    X, y = preparar_treino(treino, schema["features"])

    assert list(X.columns) == schema["features"]
    assert not X.isna().any().any()
    assert len(X) == len(y) == len(treino)

    cfg = carregar_config()
    res = executar_metodos(X, y, cfg)
    for r in res.values():
        assert set(r["ranking"]["feature"]) == set(schema["features"])
        assert len(r["top_k"]) == cfg["votacao"]["top_k"]
