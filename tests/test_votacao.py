"""Testes da votação e da tabela de votos (Dev 2, tarefa 3).

Usa rankings montados à mão, para que o resultado esperado seja verificável
de cabeça. Regra testada: top-k por método, eleita com >= votos_minimos,
desempate por rank médio, mínimo de features completado pelo rank médio.

Rodar a partir da raiz:  python -m pytest -q tests/test_votacao.py
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (METODOS, MOTIVO_MINIMO, MOTIVO_VOTOS, executar_metodos,
                          resumir_votacao, votar)

FEATURES = [f"f{i}" for i in range(1, 9)]  # f1 ... f8


def _resultado(metodo: str, ordem: list[str], top_k: int) -> dict:
    """Cria o dict que um método devolveria, dado o ranking em ordem (melhor 1º)."""
    rk = pd.DataFrame({"feature": ordem, "score": np.arange(len(ordem), 0, -1, dtype=float)})
    rk["rank"] = np.arange(1, len(ordem) + 1)
    return {"metodo": metodo, "params": {}, "ranking": rk, "top_k": ordem[:top_k]}


def _regra(top_k=3, votos_minimos=2, minimo_features=2):
    return {"metodos": list(METODOS), "top_k": top_k, "votos_minimos": votos_minimos,
            "desempate": "rank_medio", "minimo_features": minimo_features}


@pytest.fixture
def resultados():
    # top-3 de cada método:
    #   kbest : f1 f2 f3        rf: f1 f2 f4        rfe: f1 f3 f5
    # votos: f1=3, f2=2, f3=2, f4=1, f5=1, f6..f8=0
    return {
        "selectkbest_f_classif": _resultado("selectkbest_f_classif",
                                            ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"], 3),
        "random_forest": _resultado("random_forest",
                                    ["f1", "f2", "f4", "f3", "f5", "f6", "f7", "f8"], 3),
        "rfe_logistic": _resultado("rfe_logistic",
                                   ["f1", "f3", "f5", "f2", "f4", "f6", "f7", "f8"], 3),
    }


def test_contagem_de_votos_e_eleitas(resultados):
    tabela, eleitas = votar(resultados, _regra())
    votos = tabela.set_index("feature")["total_votos"]
    assert votos["f1"] == 3 and votos["f2"] == 2 and votos["f3"] == 2
    assert votos["f4"] == 1 and votos["f5"] == 1 and votos["f8"] == 0
    assert eleitas == ["f1", "f2", "f3"]
    assert (tabela.loc[tabela["eleita"], "motivo"] == MOTIVO_VOTOS).all()
    assert (tabela.loc[~tabela["eleita"], "motivo"] == "").all()


def test_rank_medio_e_desempate_na_ordem(resultados):
    tabela, eleitas = votar(resultados, _regra())
    rm = tabela.set_index("feature")["rank_medio"]
    assert rm["f1"] == 1.0
    assert rm["f2"] == pytest.approx((2 + 2 + 4) / 3, abs=1e-3)
    assert rm["f3"] == pytest.approx((3 + 4 + 2) / 3, abs=1e-3)
    # f2 e f3 empatam em votos (2); f2 vem antes por rank médio menor
    assert eleitas.index("f2") < eleitas.index("f3")


def test_completa_ate_o_minimo_pelo_rank_medio(resultados):
    # exige 5 features: f1, f2, f3 pelos votos + 2 completadas pelo rank médio
    tabela, eleitas = votar(resultados, _regra(minimo_features=5))
    assert len(eleitas) == 5
    completadas = tabela.loc[tabela["motivo"] == MOTIVO_MINIMO, "feature"].tolist()
    # f4: (4+3+5)/3 = 4.0 ; f5: (5+5+3)/3 = 4.33 ; f6: 6.0 -> entram f4 e f5
    assert sorted(completadas) == ["f4", "f5"]
    assert eleitas[:3] == ["f1", "f2", "f3"]
    assert eleitas[3:] == ["f4", "f5"]


def test_formato_e_ordem_da_tabela(resultados):
    regra = _regra()
    tabela, _ = votar(resultados, regra)
    esperadas = (["feature"] + [f"voto_{m}" for m in regra["metodos"]] + ["total_votos"]
                 + [f"rank_{m}" for m in regra["metodos"]] + ["rank_medio", "eleita", "motivo"])
    assert list(tabela.columns) == esperadas
    assert len(tabela) == len(FEATURES) and set(tabela["feature"]) == set(FEATURES)
    assert tabela["eleita"].dtype == bool
    # eleitas primeiro, depois por votos decrescentes, depois rank médio crescente
    assert tabela["eleita"].tolist() == sorted(tabela["eleita"], reverse=True)
    nao = tabela.loc[~tabela["eleita"]]
    assert nao["total_votos"].tolist() == sorted(nao["total_votos"], reverse=True)


def test_votacao_e_deterministica(resultados):
    t1, e1 = votar(resultados, _regra())
    t2, e2 = votar(resultados, _regra())
    pd.testing.assert_frame_equal(t1, t2)
    assert e1 == e2


def test_resumo(resultados):
    regra = _regra(minimo_features=5)
    tabela, _ = votar(resultados, regra)
    r = resumir_votacao(tabela, regra)
    assert r["n_features_avaliadas"] == 8
    assert r["n_eleitas"] == 5
    assert r["n_eleitas_por_votos"] == 3
    assert r["n_completadas_pelo_minimo"] == 2
    assert (r["n_com_3_votos"], r["n_com_2_votos"], r["n_com_1_voto"], r["n_sem_votos"]) == (1, 2, 2, 3)


def test_metodo_faltando_e_conjunto_diferente_dao_erro(resultados):
    with pytest.raises(ValueError, match="faltam resultados"):
        votar({k: v for k, v in resultados.items() if k != "rfe_logistic"}, _regra())
    quebrado = dict(resultados)
    quebrado["rfe_logistic"] = _resultado("rfe_logistic", ["f1", "f2", "f3"], 3)
    with pytest.raises(ValueError, match="conjunto diferente"):
        votar(quebrado, _regra())


# --------------------------------------------------------------------------- #
# Integração: métodos reais em dados sintéticos -> votação respeita a regra
# --------------------------------------------------------------------------- #
def test_fluxo_metodos_mais_votacao():
    from sklearn.datasets import make_classification

    from src.ingestao import carregar_config
    X, y = make_classification(n_samples=500, n_features=12, n_informative=4,
                               n_redundant=0, n_clusters_per_class=1,
                               weights=[0.8, 0.2], shuffle=False, random_state=42)
    X = pd.DataFrame(X, columns=[f"c{i}" for i in range(12)])
    cfg = carregar_config()
    cfg["votacao"].update({"top_k": 5, "minimo_features": 3})
    res = executar_metodos(X, pd.Series(y), cfg)
    tabela, eleitas = votar(res, cfg=cfg)
    assert len(eleitas) >= 3
    assert (tabela.loc[tabela["motivo"] == MOTIVO_VOTOS, "total_votos"] >= 2).all()
    assert tabela[[f"voto_{m}" for m in METODOS]].sum().eq(5).all()  # cada método deu 5 votos
