"""Contrato dos JSONs do painel (Dev 5, critério 10).

1. Todo cenário mock em web/dados/mocks/ passa no schema e nas regras de coerência.
2. O validador pega quebras típicas (campo faltando, enum errado, arquivos divergentes,
   decisão que contradiz a regra de promoção).
3. Se public/data/ existir (saída do exportador), também é validada.

Rodar a partir da raiz:  python -m pytest -q tests/test_contrato_painel.py
"""
import copy

import pytest

from src.contrato_painel import (RAIZ, carregar_pasta, validar_coerencia, validar_formato,
                                 validar_pasta)

DIR_MOCKS = RAIZ / "web" / "dados" / "mocks"
CENARIOS = sorted(p.name for p in DIR_MOCKS.iterdir() if p.is_dir())


def test_existem_os_tres_cenarios():
    assert {"padrao", "mantido", "vazio"} <= set(CENARIOS)


@pytest.mark.parametrize("cenario", CENARIOS)
def test_mock_valido(cenario):
    assert validar_pasta(DIR_MOCKS / cenario) == []


def test_cenarios_cobrem_os_estados_da_visao_geral():
    padrao = carregar_pasta(DIR_MOCKS / "padrao")
    mantido = carregar_pasta(DIR_MOCKS / "mantido")
    vazio = carregar_pasta(DIR_MOCKS / "vazio")
    # banner de troca de modelo
    assert padrao["vigente"]["modelo_mudou"] and padrao["vigente"]["anterior"] is not None
    # último ciclo mantido, sem dados novos
    assert not mantido["vigente"]["modelo_mudou"]
    assert mantido["ciclos"]["ciclos"][-1]["lotes_novos"] == []
    # estado vazio (antes do 1º ciclo)
    assert vazio["vigente"]["vigente"] is None and vazio["ciclos"]["ciclos"] == []


@pytest.mark.skipif(not (RAIZ / "public/data/ciclos.json").exists(),
                    reason="public/data/ ainda não foi gerado")
def test_public_data_valido():
    assert validar_pasta(RAIZ / "public/data") == []


# --------------------------------------------------------------------------- #
# O validador precisa falhar quando o contrato é quebrado
# --------------------------------------------------------------------------- #
@pytest.fixture
def dados():
    return copy.deepcopy(carregar_pasta(DIR_MOCKS / "mantido"))


def _erros(d):
    return validar_formato(d) or validar_coerencia(d)


def test_detecta_campo_faltando(dados):
    del dados["ciclos"]["ciclos"][0]["data_version"]
    assert any("data_version" in e for e in _erros(dados))


def test_detecta_campo_extra(dados):
    dados["vigente"]["vigente"]["acuracia"] = 0.99
    assert _erros(dados)


def test_detecta_enum_invalido(dados):
    dados["ciclos"]["ciclos"][1]["decisao"] = "promovida"
    assert _erros(dados)


def test_detecta_versao_formato(dados):
    dados["votacao"]["versao_formato"] = 2
    assert _erros(dados)


def test_detecta_cycle_id_divergente(dados):
    dados["competicao"]["ciclos"][0]["cycle_id"] = "20990101-0000-r1"
    assert any("cycle_id" in e for e in _erros(dados))


def test_detecta_decisao_contra_a_regra(dados):
    ultimo = dados["ciclos"]["ciclos"][-1]          # c3: ganho 0 → mantido
    ultimo["decisao"] = "promovido"
    dados["competicao"]["ciclos"][-1]["decisao"] = "promovido"
    assert any("regra de promoção" in e for e in _erros(dados))


def test_detecta_voto_incoerente(dados):
    linha = dados["votacao"]["ciclos"][0]["tabela"][0]
    linha["votos"]["rfe_logistic"] = 0
    assert _erros(dados)


def test_detecta_modelo_sem_as_eleitas(dados):
    dados["competicao"]["ciclos"][0]["modelos"][0]["features"] = ["sent_tnx"]
    assert any("features eleitas" in e for e in _erros(dados))


def test_detecta_vigente_desatualizado(dados):
    dados["vigente"]["modelo_mudou"] = True
    assert any("modelo_mudou" in e for e in _erros(dados))


def test_detecta_origens_misturadas(dados):
    dados["previsoes"]["origem"] = "pipeline"
    assert any("origens" in e for e in _erros(dados))
