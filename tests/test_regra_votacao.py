"""A regra de votação foi fixada em config.yaml ANTES do ciclo 1 (Dev 2, tarefa 2).

Estes testes "travam" os valores combinados com o grupo: se alguém alterar a
regra depois de ver resultados, a suíte falha e a mudança fica visível na
revisão. Também cobrem a validação da seção ``votacao``.

Rodar a partir da raiz:  python -m pytest -q tests/test_regra_votacao.py
"""
import copy

import pytest

from src.features import METODOS, carregar_regra_votacao
from src.ingestao import carregar_config

REGRA_COMBINADA = {
    "metodos": ["selectkbest_f_classif", "random_forest", "rfe_logistic"],
    "top_k": 10,
    "votos_minimos": 2,
    "desempate": "rank_medio",
    "minimo_features": 5,
}


def test_regra_do_config_e_a_combinada_com_o_grupo():
    assert carregar_regra_votacao() == REGRA_COMBINADA


def test_regra_coerente_com_os_metodos_implementados():
    regra = carregar_regra_votacao()
    assert set(regra["metodos"]) == set(METODOS)
    assert regra["votos_minimos"] <= len(regra["metodos"])
    assert regra["minimo_features"] <= regra["top_k"]


def _cfg_com(**alteracoes):
    cfg = copy.deepcopy(carregar_config())
    cfg["votacao"].update(alteracoes)
    return cfg


@pytest.mark.parametrize("alteracao, erro", [
    ({"metodos": ["random_forest"]}, "exatamente 3"),
    ({"metodos": ["random_forest", "random_forest", "rfe_logistic"]}, "exatamente 3"),
    ({"metodos": ["pca", "random_forest", "rfe_logistic"]}, "desconhecidos"),
    ({"top_k": 0}, "top_k"),
    ({"votos_minimos": 4}, "votos_minimos"),
    ({"votos_minimos": 0}, "votos_minimos"),
    ({"minimo_features": 11}, "minimo_features"),
    ({"desempate": "aleatorio"}, "desempate"),
])
def test_valores_invalidos_sao_rejeitados(alteracao, erro):
    with pytest.raises(ValueError, match=erro):
        carregar_regra_votacao(_cfg_com(**alteracao))


def test_campo_faltando_e_rejeitado():
    cfg = copy.deepcopy(carregar_config())
    del cfg["votacao"]["votos_minimos"]
    with pytest.raises(ValueError, match="votos_minimos"):
        carregar_regra_votacao(cfg)


def test_sem_secao_votacao_e_rejeitado():
    cfg = copy.deepcopy(carregar_config())
    del cfg["votacao"]
    with pytest.raises(ValueError, match="votacao"):
        carregar_regra_votacao(cfg)
