"""Registro da votação no MLflow e artefatos do ciclo (Dev 2, tarefa 4).

Usa um MLflow SQLite temporário e uma pasta de artefatos temporária, então
não toca em mlflow.db nem em artifacts/ do repositório.

Rodar a partir da raiz:  python -m pytest -q tests/test_registro_mlflow.py
"""
import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (ARQ_ELEITAS, ARQ_TABELA, ETAPA, METODOS,
                          carregar_features_eleitas, obter_cycle_id,
                          registrar_votacao, tracking_uri, votar)
from src.ingestao import carregar_config


def _resultado(metodo, ordem, top_k):
    rk = pd.DataFrame({"feature": ordem, "score": np.arange(len(ordem), 0, -1, dtype=float)})
    rk["rank"] = np.arange(1, len(ordem) + 1)
    return {"metodo": metodo, "params": {"p": 1, "seed": 42}, "ranking": rk, "top_k": ordem[:top_k]}


@pytest.fixture
def ambiente(tmp_path: Path, monkeypatch):
    """config apontando MLflow e artifacts para pastas temporárias."""
    cfg = copy.deepcopy(carregar_config())
    cfg["projeto"]["tracking_uri"] = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    cfg["projeto"]["experimento_mlflow"] = "teste_votacao"
    cfg["votacao"].update({"top_k": 3, "minimo_features": 2})
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("CYCLE_ID", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    return cfg, tmp_path


@pytest.fixture
def votacao(ambiente):
    cfg, _ = ambiente
    feats = ["f1", "f2", "f3", "f4", "f5", "f6"]
    res = {
        "selectkbest_f_classif": _resultado("selectkbest_f_classif", feats, 3),
        "random_forest": _resultado("random_forest", ["f1", "f2", "f4", "f3", "f5", "f6"], 3),
        "rfe_logistic": _resultado("rfe_logistic", ["f1", "f3", "f5", "f2", "f4", "f6"], 3),
    }
    regra = {"metodos": list(METODOS), "top_k": 3, "votos_minimos": 2,
             "desempate": "rank_medio", "minimo_features": 2}
    tabela, eleitas = votar(res, regra)
    return res, tabela, eleitas, regra


def test_cycle_id_vem_da_variavel_ou_e_local(monkeypatch):
    monkeypatch.setenv("CYCLE_ID", "20261110-0900-r12")
    assert obter_cycle_id() == "20261110-0900-r12"
    monkeypatch.delenv("CYCLE_ID")
    assert obter_cycle_id().endswith("-local")


def test_tracking_uri_sqlite_relativo_vira_absoluto(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = copy.deepcopy(carregar_config())
    cfg["projeto"]["tracking_uri"] = "sqlite:///mlflow.db"
    uri = tracking_uri(cfg)
    assert uri.startswith("sqlite:///") and uri.endswith("/mlflow.db")
    assert Path(uri[len("sqlite:///"):]).is_absolute()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert tracking_uri(cfg) == "http://mlflow:5000"


def test_registra_artefatos_e_run(ambiente, votacao):
    cfg, tmp = ambiente
    res, tabela, eleitas, regra = votacao
    saida = tmp / "artifacts" / "ciclo-teste"

    reg = registrar_votacao(res, tabela, eleitas, regra, data_version="v1-abcd1234",
                            cycle_id="ciclo-teste", cfg=cfg, dir_saida=saida)

    # --- arquivos no disco -------------------------------------------------
    assert (saida / ARQ_ELEITAS).exists() and (saida / ARQ_TABELA).exists()
    for m in METODOS:
        assert (saida / "rankings" / f"{m}.csv").exists()
    js = json.loads((saida / ARQ_ELEITAS).read_text("utf-8"))
    assert js["cycle_id"] == "ciclo-teste" and js["data_version"] == "v1-abcd1234"
    assert js["features"] == eleitas == ["f1", "f2", "f3"]
    assert js["regra"] == regra and js["mlflow_run_id"] == reg["run_id"]
    assert set(js["top_k_por_metodo"]) == set(METODOS)
    assert set(js["motivo"]) == set(eleitas)
    tab = pd.read_csv(saida / ARQ_TABELA)
    assert list(tab["feature"]) == list(tabela["feature"])

    # --- run no MLflow -----------------------------------------------------
    import mlflow
    mlflow.set_tracking_uri(cfg["projeto"]["tracking_uri"])
    run = mlflow.get_run(reg["run_id"])
    assert run.data.tags["cycle_id"] == "ciclo-teste"
    assert run.data.tags["data_version"] == "v1-abcd1234"
    assert run.data.tags["etapa"] == ETAPA
    assert run.data.params["regra.top_k"] == "3"
    assert run.data.params["regra.votos_minimos"] == "2"
    assert run.data.params["random_forest.seed"] == "42"
    assert run.data.params["features_eleitas"] == "f1,f2,f3"
    assert run.data.metrics["n_eleitas"] == 3
    assert run.data.metrics["n_com_3_votos"] == 1
    artefatos = {a.path for a in mlflow.artifacts.list_artifacts(run_id=reg["run_id"])}
    assert {ARQ_ELEITAS, ARQ_TABELA, "rankings"} <= artefatos


def test_carregar_features_eleitas_por_ciclo_e_mais_recente(ambiente, votacao, monkeypatch):
    cfg, tmp = ambiente
    cfg["votacao"]["dir_artifacts"] = str(tmp / "artifacts")   # absoluto: RAIZ / abs == abs
    res, tabela, eleitas, regra = votacao
    base = tmp / "artifacts"
    registrar_votacao(res, tabela, eleitas, regra, "v1-aaaa0000", "c1", cfg, base / "c1")
    (base / "c1" / ARQ_ELEITAS).touch()
    import time; time.sleep(0.05)
    registrar_votacao(res, tabela, eleitas, regra, "v2-bbbb0000", "c2", cfg, base / "c2")

    assert carregar_features_eleitas("c1", cfg)["data_version"] == "v1-aaaa0000"
    assert carregar_features_eleitas(cfg=cfg)["cycle_id"] == "c2"       # mais recente
    monkeypatch.setenv("CYCLE_ID", "c1")
    assert carregar_features_eleitas(cfg=cfg)["cycle_id"] == "c1"       # pela variável
    with pytest.raises(FileNotFoundError):
        carregar_features_eleitas("nao-existe", cfg)
