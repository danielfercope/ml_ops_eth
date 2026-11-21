"""cycle_id, tags do ciclo e relocalização do mlflow.db (Dev 4).

Rodar a partir da raiz:  python -m pytest -q tests/test_rastreio.py
"""
import copy
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src import rastreio
from src.ingestao import RAIZ, carregar_config


def test_gerar_cycle_id_formato():
    agora = datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc)
    assert rastreio.gerar_cycle_id(12, agora) == "20261110-0900-r12"
    assert rastreio.gerar_cycle_id(None, agora) == "20261110-0900-local"
    assert rastreio.gerar_cycle_id("", agora) == "20261110-0900-local"


def test_obter_cycle_id_usa_variavel_ou_fallback_local(monkeypatch):
    monkeypatch.setenv("CYCLE_ID", " 20261110-0900-r12 ")
    assert rastreio.obter_cycle_id() == "20261110-0900-r12"
    monkeypatch.delenv("CYCLE_ID")
    cid = rastreio.obter_cycle_id()
    assert cid.endswith("-local") and rastreio.eh_local(cid)


def test_features_usa_o_cycle_id_do_rastreio(monkeypatch):
    from src.features import obter_cycle_id

    monkeypatch.setenv("CYCLE_ID", "20261110-0900-r7")
    assert obter_cycle_id() == "20261110-0900-r7"


def test_tracking_uri_absoluto_e_variavel_tem_prioridade(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    uri = rastreio.tracking_uri()
    assert uri.startswith("sqlite:///")
    assert Path(uri[len("sqlite:///"):]).is_absolute()
    assert Path(uri[len("sqlite:///"):]).parent == RAIZ.resolve()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:////tmp/outro.db")
    assert rastreio.tracking_uri() == "sqlite:////tmp/outro.db"


def test_tags_do_ciclo(monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    tags = rastreio.tags_do_ciclo("previsoes", "c1", "v1-abc", algoritmo="xgboost", vazio=None)
    assert tags == {"cycle_id": "c1", "data_version": "v1-abc", "etapa": "previsoes",
                    "gatilho": "schedule", "github_run_id": "123", "algoritmo": "xgboost"}
    with pytest.raises(ValueError):
        rastreio.tags_do_ciclo("etapa_inventada", "c1", "v1-abc")


def test_iniciar_run_grava_tags_e_artefatos_em_mlartifacts(tmp_path, monkeypatch):
    cfg = copy.deepcopy(carregar_config())
    cfg["projeto"]["tracking_uri"] = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    cfg["projeto"]["experimento_mlflow"] = "teste_rastreio"
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert rastreio.dir_artefatos_mlflow(cfg) == (tmp_path / "mlartifacts").resolve()

    with rastreio.iniciar_run("etl", cycle_id="c9", data_version="v1-x", cfg=cfg) as run:
        pass
    import mlflow

    r = mlflow.get_run(run.info.run_id)
    assert {k: r.data.tags[k] for k in rastreio.TAGS_OBRIGATORIAS} == \
        {"cycle_id": "c9", "data_version": "v1-x", "etapa": "etl"}
    assert r.info.artifact_uri.startswith((tmp_path / "mlartifacts").resolve().as_uri())
    assert len(rastreio.runs_do_ciclo("c9", cfg)) == 1


def _db_com_caminhos(p: Path, raiz: str) -> None:
    with sqlite3.connect(p) as con:
        con.execute("CREATE TABLE experiments (experiment_id INTEGER, artifact_location VARCHAR(256))")
        con.execute("CREATE TABLE runs (run_uuid VARCHAR(32), artifact_uri VARCHAR(200), status VARCHAR(9))")
        con.execute("CREATE TABLE logged_models (model_id VARCHAR, artifact_location TEXT)")
        con.executemany("INSERT INTO experiments VALUES (?, ?)", [
            (0, f"{raiz}/mlruns/0"), (1, f"file://{raiz}/mlartifacts")])
        con.execute("INSERT INTO runs VALUES ('r1', ?, 'FINISHED')", (f"file://{raiz}/mlartifacts/r1/artifacts",))
        con.execute("INSERT INTO runs VALUES ('r2', ?, 'FINISHED')", (f"file://{raiz}-outro/x",))
        con.execute("INSERT INTO logged_models VALUES ('m1', ?)", (f"file://{raiz}/mlartifacts/models/m1/artifacts",))


def test_relocalizar_troca_a_raiz_antiga_pela_pasta_do_banco(tmp_path):
    db = tmp_path / "mlflow.db"
    antiga = "/home/runner/work/ml_ops_eth/ml_ops_eth"
    _db_com_caminhos(db, antiga)

    r = rastreio.relocalizar_artefatos(db=db)
    assert r["antigas"] == [antiga] and r["linhas"] == 4
    atual = tmp_path.resolve().as_posix()   # a pasta onde o banco está agora
    with sqlite3.connect(db) as con:
        exps = dict(con.execute("SELECT experiment_id, artifact_location FROM experiments"))
        runs = dict(con.execute("SELECT run_uuid, artifact_uri FROM runs"))
        (modelo,) = con.execute("SELECT artifact_location FROM logged_models").fetchone()
    assert exps == {0: f"{atual}/mlruns/0", 1: f"file://{atual}/mlartifacts"}
    assert runs["r1"] == f"file://{atual}/mlartifacts/r1/artifacts"
    assert runs["r2"] == f"file://{antiga}-outro/x"       # prefixo parecido não é tocado
    assert modelo == f"file://{atual}/mlartifacts/models/m1/artifacts"

    # idempotente
    assert rastreio.relocalizar_artefatos(db=db) == {"antigas": [], "linhas": 0}


def test_relocalizar_sem_banco_nao_faz_nada(tmp_path):
    assert rastreio.relocalizar_artefatos(db=tmp_path / "nao_existe.db") == {"antigas": [], "linhas": 0}
