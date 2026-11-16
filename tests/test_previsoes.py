"""Previsões do vigente e histórico entre ciclos (Dev 4, notebook 05).

Usa arquivos temporários: não lê o manifesto nem o histórico reais.
Rodar a partir da raiz:  python -m pytest -q tests/test_previsoes.py
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src import previsoes as pv
from src.ingestao import carregar_config


class ModeloFixo:
    """Probabilidade = valor da coluna 'p' (para testar sem treinar nada)."""

    def predict_proba(self, df):
        p = df["p"].to_numpy(dtype=float)
        return np.column_stack([1 - p, p])


@pytest.fixture
def df():
    return pd.DataFrame({"address": [f"0x{i}" for i in range(6)],
                         "p": [0.9, 0.6, 0.4, 0.1, 0.5, 0.49],
                         "flag": [1, 0, 1, 0, 1, 0]})


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(carregar_config())
    c["modelos"] = {**c.get("modelos", {}), "limiar_classificacao": 0.5}
    c["promocao"] = {**c.get("promocao", {}),
                     "manifesto": str(tmp_path / "models" / "modelo_vigente.json"),
                     "historico": str(tmp_path / "models" / "historico_decisoes.csv")}
    c["votacao"] = {**c["votacao"], "dir_artifacts": str(tmp_path / "artifacts")}
    c["dados"] = {**c["dados"], "dir_processed": str(tmp_path / "processed")}
    return c


def test_prever_aplica_o_limiar_e_traz_o_rotulo(df, cfg):
    prev = pv.prever(ModeloFixo(), df, cfg)
    assert list(prev.columns) == ["address", "probabilidade_fraude", "previsao", "rotulo_real", "acerto"]
    assert prev["previsao"].tolist() == [1, 1, 0, 0, 1, 0]       # 0.5 é fraude (>=)
    r = pv.resumir(prev)
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 1, 1, 2)
    assert r["f1"] == pytest.approx(2 / 3)


def test_amostra_e_estavel_e_balanceada(df, cfg):
    prev = pv.prever(ModeloFixo(), df, cfg)
    a1, a2 = pv.amostra(prev, n=4, seed=1), pv.amostra(prev, n=4, seed=1)
    assert a1.equals(a2)
    assert a1["rotulo_real"].value_counts().to_dict() == {1: 2, 0: 2}


def test_sem_manifesto_nao_ha_vigente(cfg):
    assert pv.carregar_manifesto(cfg) is None
    assert pv.carregar_vigente(cfg) == (None, None)


def test_historico_ciclos_marca_o_que_mudou(cfg, tmp_path):
    (tmp_path / "models").mkdir()
    pd.DataFrame([
        {"cycle_id": "c1", "data_version": "v1-a", "vigente_run_id": "r1", "decisao": "promovido"},
        {"cycle_id": "c2", "data_version": "v2-b", "vigente_run_id": "r2", "decisao": "promovido"},
        {"cycle_id": "c3", "data_version": "v2-b", "vigente_run_id": "r2", "decisao": "mantido"},
    ]).to_csv(tmp_path / "models" / "historico_decisoes.csv", index=False)
    for cid, feats in {"c1": ["a", "b"], "c2": ["a", "c"], "c3": ["c", "a"]}.items():
        (tmp_path / "artifacts" / cid).mkdir(parents=True)
        (tmp_path / "artifacts" / cid / "features_eleitas.json").write_text(json.dumps({"features": feats}))
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "lotes.json").write_text(json.dumps({"lotes": [
        {"nome": "lote_01", "cycle_id": "c1"}, {"nome": "lote_02", "cycle_id": "c2"}]}))

    h = pv.historico_ciclos(cfg)
    assert h["lotes_novos"].tolist() == ["lote_01", "lote_02", ""]
    assert h["mudou_data_version"].tolist() == [True, True, False]
    assert h["mudou_features"].tolist() == [True, True, False]
    assert h["mudou_modelo"].tolist() == [True, True, False]
    assert pv.diff_features(["a", "b"], ["a", "c"]) == {"entraram": ["c"], "sairam": ["b"], "mantidas": ["a"]}


def test_gravar_previsoes(df, cfg, tmp_path):
    prev = pv.prever(ModeloFixo(), df, cfg)
    man = {"run_id": "r1", "modelo": "B", "algoritmo": "xgboost", "features": ["a"]}
    caminhos = pv.gravar_previsoes(prev, pv.resumir(prev), man, "c1", "v1-a", cfg)
    assert caminhos["previsoes"] == tmp_path / "artifacts" / "c1" / "previsoes.csv"
    resumo = json.loads(caminhos["resumo"].read_text())
    assert resumo["vigente"]["run_id"] == "r1" and resumo["particao"] == "teste"
    assert len(pd.read_csv(caminhos["previsoes"])) == len(df)
