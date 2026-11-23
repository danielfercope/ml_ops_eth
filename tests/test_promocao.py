"""Regra de promoção, escolha do vencedor, manifesto e histórico (Dev 3).

Rodar a partir da raiz:  python -m pytest -q tests/test_promocao.py
"""
import copy
import json

import pandas as pd
import pytest

from src.ingestao import carregar_config
from src.promocao import (COLUNAS_HISTORICO, MANTIDO, PROMOVIDO, carregar_manifesto,
                          carregar_regra_promocao, decidir, escolher_vencedor,
                          gravar_decisao, linha_historico)

REGRA = {"criterio_vencedor": "f1_validacao", "desempate_vencedor": "A",
         "margem_minima": 0.01, "f1_minimo": 0.60}


def _cand(f1a, f1b):
    return {"A": {"algoritmo": "logistic_regression", "metricas_validacao": {"f1": f1a}},
            "B": {"algoritmo": "xgboost", "metricas_validacao": {"f1": f1b}}}


# --------------------------------------------------------------------------- #
# A regra está no config e não mudou depois de ver resultados
# --------------------------------------------------------------------------- #
def test_regra_fixada_no_config():
    regra = carregar_regra_promocao()
    assert regra == REGRA
    cfg = carregar_config()
    assert cfg["metrica"]["principal"] == "f1"
    assert cfg["modelos"]["A"] == "logistic_regression"
    assert cfg["modelos"]["B"] == "xgboost"


def test_regra_invalida_falha():
    cfg = copy.deepcopy(carregar_config())
    cfg["promocao"]["f1_minimo"] = 1.5
    with pytest.raises(ValueError):
        carregar_regra_promocao(cfg)
    del cfg["promocao"]["margem_minima"]
    with pytest.raises(ValueError):
        carregar_regra_promocao(cfg)


# --------------------------------------------------------------------------- #
# Vencedor do ciclo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("f1a,f1b,esperado", [
    (0.80, 0.90, "B"),
    (0.91, 0.90, "A"),
    (0.85, 0.85, "A"),          # empate -> A (desempate do config)
])
def test_escolher_vencedor(f1a, f1b, esperado):
    assert escolher_vencedor(_cand(f1a, f1b), REGRA) == esperado


def test_desempate_configuravel():
    assert escolher_vencedor(_cand(0.85, 0.85), {**REGRA, "desempate_vencedor": "B"}) == "B"


# --------------------------------------------------------------------------- #
# Decisão
# --------------------------------------------------------------------------- #
def test_ciclo_1_sem_vigente_promove():
    d = decidir(0.85, None, REGRA)
    assert d["decisao"] == PROMOVIDO and "não há modelo vigente" in d["motivo"]


def test_ciclo_1_abaixo_do_minimo_nao_promove():
    d = decidir(0.55, None, REGRA)
    assert d["decisao"] == MANTIDO and "mínimo" in d["motivo"]


@pytest.mark.parametrize("novo,vigente,decisao", [
    (0.90, 0.85, PROMOVIDO),    # supera com folga
    (0.86, 0.85, PROMOVIDO),    # exatamente vigente + 0,01 (ponto flutuante não atrapalha)
    (0.71, 0.70, PROMOVIDO),
    (0.859, 0.85, MANTIDO),     # melhor, mas não pela margem
    (0.85, 0.85, MANTIDO),      # ciclo 3 sem dados novos: mesmo F1 -> mantido
    (0.80, 0.85, MANTIDO),      # pior
    (0.59, 0.40, MANTIDO),      # supera a margem mas fica abaixo de 0,60
])
def test_regra_de_promocao(novo, vigente, decisao):
    d = decidir(novo, vigente, REGRA)
    assert d["decisao"] == decisao
    assert d["f1_necessario"] == pytest.approx(max(vigente + 0.01, 0.60))


def test_motivo_explica_a_falha():
    d = decidir(0.859, 0.85, REGRA)
    assert "margem" in d["motivo"] and "0.8600" in d["motivo"]
    d = decidir(0.59, 0.40, REGRA)
    assert "mínimo" in d["motivo"]


# --------------------------------------------------------------------------- #
# Manifesto e histórico
# --------------------------------------------------------------------------- #
def _manifesto(cycle_id, decisao, run_id="r1"):
    return {
        "cycle_id": cycle_id, "data_version": "v1-aaaa0000", "decisao": decisao,
        "motivo": "teste", "decidido_em": "2026-11-10T09:00:00+00:00",
        "run_id": run_id, "model_uri": f"runs:/{run_id}/modelo", "modelo": "B",
        "algoritmo": "xgboost", "features": ["f1", "f2"], "n_features": 2,
        "f1_validacao": 0.9, "f1_teste": 0.88, "metricas_validacao": {"f1": 0.9},
        "metricas_teste": {"f1": 0.88}, "treinado_no_ciclo": cycle_id,
        "treinado_com_data_version": "v1-aaaa0000", "promovido_no_ciclo": cycle_id,
        "promovido_em": "2026-11-10T09:00:00+00:00", "registry": None,
        "desafiante": {"modelo": "B", "algoritmo": "xgboost", "run_id": run_id, "f1_validacao": 0.9},
        "candidatos": {"A": {"algoritmo": "logistic_regression", "run_id": "ra", "f1_validacao": 0.8},
                       "B": {"algoritmo": "xgboost", "run_id": run_id, "f1_validacao": 0.9}},
        "vigente_anterior": None, "f1_necessario": 0.6, "regra": REGRA, "decisao_run_id": "d1",
    }


def test_grava_manifesto_historico_e_decisao(tmp_path):
    man, hist, ciclo = tmp_path / "models/modelo_vigente.json", tmp_path / "models/hist.csv", tmp_path / "c1"
    m = _manifesto("c1", PROMOVIDO)
    gravar_decisao({"manifesto": m, "linha_historico": linha_historico(m)},
                   caminho_man=man, caminho_hist=hist, dir_ciclo=ciclo)

    lido = json.loads(man.read_text("utf-8"))
    for campo in ("cycle_id", "data_version", "run_id", "algoritmo", "features",
                  "f1_validacao", "f1_teste", "decisao", "motivo"):
        assert campo in lido
    assert carregar_manifesto(caminho=man)["run_id"] == "r1"
    assert (ciclo / "decisao.json").exists()
    h = pd.read_csv(hist)
    assert list(h.columns) == COLUNAS_HISTORICO and len(h) == 1

    # ciclo 2 acrescenta uma linha; repetir o ciclo 2 não duplica
    m2 = _manifesto("c2", MANTIDO)
    for _ in range(2):
        gravar_decisao({"manifesto": m2, "linha_historico": linha_historico(m2)},
                       caminho_man=man, caminho_hist=hist, dir_ciclo=tmp_path / "c2")
    h = pd.read_csv(hist)
    assert list(h["cycle_id"]) == ["c1", "c2"]
    assert list(h["decisao"]) == [PROMOVIDO, MANTIDO]


def test_manifesto_sem_vigente_conta_como_ausente(tmp_path):
    p = tmp_path / "m.json"
    assert carregar_manifesto(caminho=p) is None
    p.write_text(json.dumps({"run_id": None, "decisao": MANTIDO}), "utf-8")
    assert carregar_manifesto(caminho=p) is None


# --------------------------------------------------------------------------- #
# Verificações de qualidade do vencedor (além do limiar de F1)
# --------------------------------------------------------------------------- #
from src import promocao  # noqa: E402
from src.promocao import (carregar_verificacoes_qualidade, checar_metricas,  # noqa: E402
                          verificar_qualidade)

VERIF = {"precision_minima": 0.50, "recall_minimo": 0.50, "taxa_prevista_min": 0.01,
         "taxa_prevista_max": 0.99, "recarregar_do_mlflow": True, "tolerancia_recarga": 1e-6}


def _met(precision=0.89, recall=0.94, f1=0.9159, tp=256, fp=32, fn=15, tn=1003):
    return {"f1": f1, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": tp + fp + fn + tn}


def _falhas(met):
    return [c["nome"] for c in checar_metricas(met, VERIF) if not c["ok"]]


def test_verificacoes_fixadas_no_config():
    assert carregar_verificacoes_qualidade() == VERIF


def test_verificacoes_invalidas_falham():
    cfg = copy.deepcopy(carregar_config())
    cfg["promocao"]["verificacoes_qualidade"]["recall_minimo"] = 1.2
    with pytest.raises(ValueError):
        carregar_verificacoes_qualidade(cfg)
    cfg = copy.deepcopy(carregar_config())
    cfg["promocao"]["verificacoes_qualidade"]["taxa_prevista_min"] = 0.995
    with pytest.raises(ValueError):
        carregar_verificacoes_qualidade(cfg)
    cfg = copy.deepcopy(carregar_config())
    del cfg["promocao"]["verificacoes_qualidade"]
    with pytest.raises(ValueError):
        carregar_verificacoes_qualidade(cfg)


def test_modelo_saudavel_passa():
    assert _falhas(_met()) == []


def test_precision_ou_recall_abaixo_do_piso():
    assert _falhas(_met(precision=0.40)) == ["precision_minima"]
    assert _falhas(_met(recall=0.30)) == ["recall_minimo"]


def test_modelo_degenerado_que_preve_uma_classe_so():
    # tudo previsto como fraude: recall 1, precision = prevalência, taxa prevista 1.0
    tudo_fraude = _met(precision=0.55, recall=1.0, tp=271, fp=222, fn=0, tn=0)
    assert "taxa_prevista" in _falhas(tudo_fraude)


def test_metricas_nao_finitas():
    assert _falhas(_met(f1=float("nan"))) == ["metricas_finitas"]


def test_recarga_do_mlflow(monkeypatch):
    cand = {"run_id": "r1", "metricas_validacao": _met()}
    monkeypatch.setattr(promocao, "carregar_modelo", lambda run_id, cfg=None: object())
    monkeypatch.setattr(promocao, "avaliar", lambda modelo, df, cfg=None: {"f1": 0.9159})
    assert verificar_qualidade(cand, pd.DataFrame(), verif=VERIF)["aprovada"]

    monkeypatch.setattr(promocao, "avaliar", lambda modelo, df, cfg=None: {"f1": 0.80})
    q = verificar_qualidade(cand, pd.DataFrame(), verif=VERIF)
    assert not q["aprovada"] and "recarregado" in q["falhas"][0]

    def falha(run_id, cfg=None):
        raise OSError("artefato ausente")
    monkeypatch.setattr(promocao, "carregar_modelo", falha)
    q = verificar_qualidade(cand, pd.DataFrame(), verif=VERIF)
    assert not q["aprovada"] and "artefato ausente" in q["falhas"][0]


def test_sem_validacao_nao_recarrega():
    q = verificar_qualidade({"run_id": "r1", "metricas_validacao": _met()}, None, verif=VERIF)
    assert q["aprovada"] and "recarga_mlflow" not in [c["nome"] for c in q["checagens"]]


@pytest.mark.parametrize("vigente", [None, 0.85])
def test_qualidade_reprovada_impede_promocao(vigente):
    reprovada = {"aprovada": False, "falhas": ["recall 0.3000 < mínimo 0.50"]}
    d = decidir(0.90, vigente, REGRA, reprovada)
    assert d["decisao"] == MANTIDO
    assert "verificações de qualidade" in d["motivo"] and "recall" in d["motivo"]
    # aprovada: segue a regra de F1 normalmente
    assert decidir(0.90, vigente, REGRA, {"aprovada": True, "falhas": []})["decisao"] == PROMOVIDO


def test_qualidade_nao_salva_quem_falha_no_limiar():
    d = decidir(0.855, 0.85, REGRA, {"aprovada": True, "falhas": []})
    assert d["decisao"] == MANTIDO and "margem" in d["motivo"]


def test_historico_registra_a_qualidade():
    m = _manifesto("c1", MANTIDO)
    m["qualidade"] = {"aprovada": False, "falhas": ["a", "b"], "checagens": [], "limites": VERIF}
    linha = linha_historico(m)
    assert linha["qualidade_aprovada"] is False and linha["qualidade_falhas"] == "a; b"
