"""Critérios de 'pronto' do Dev 1.

1. Rodar o ETL duas vezes sobre os mesmos lotes -> mesmo data_version e mesmas partições.
2. Adicionar o lote_02 não move nenhum registro antigo de partição.
3. Sem sobreposição de endereços entre treino / validação / teste.
4. Pré-processador ajustado no treino devolve DataFrame com os nomes das features.

Rodar a partir da raiz:  python -m pytest -q tests
(usa uma cópia temporária dos lotes; não altera data/ do repositório)
"""
import copy
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.ingestao import RAIZ, carregar_config
from src.preprocessamento import (ALVO, ID, PARTICOES, carregar_particoes,
                                  executar_etl, montar_preprocessador)

LOTE1 = RAIZ / "data/raw/lote_01.csv"
LOTE2 = RAIZ / "data/reserva/lote_02.csv"
if not LOTE2.exists():
    LOTE2 = RAIZ / "data/raw/lote_02.csv"

pytestmark = pytest.mark.skipif(not (LOTE1.exists() and LOTE2.exists()),
                                reason="rode `python -m src.ingestao gerar-lotes` antes")


@pytest.fixture
def cfg_tmp(tmp_path: Path):
    cfg = copy.deepcopy(carregar_config())
    for chave in ("dir_raw", "dir_reserva", "dir_processed"):
        (tmp_path / chave).mkdir()
        cfg["dados"][chave] = str(tmp_path / chave)
    shutil.copy(LOTE1, tmp_path / "dir_raw" / "lote_01.csv")
    return cfg, tmp_path


def _enderecos(cfg):
    return {n: set(p[ID]) for n, p in zip(PARTICOES, carregar_particoes(cfg))}


def test_etl_deterministico(cfg_tmp):
    cfg, _ = cfg_tmp
    r1 = executar_etl(cfg)
    e1 = _enderecos(cfg)
    r2 = executar_etl(cfg)
    assert r1["data_version"] == r2["data_version"]
    assert r1["data_version"].startswith("v1-")
    assert r1["particoes"] == r2["particoes"]
    assert e1 == _enderecos(cfg)


def test_lote2_nao_move_registros_antigos(cfg_tmp):
    cfg, tmp = cfg_tmp
    r1 = executar_etl(cfg)
    antes = _enderecos(cfg)

    shutil.copy(LOTE2, tmp / "dir_raw" / "lote_02.csv")
    r2 = executar_etl(cfg)
    depois = _enderecos(cfg)

    assert r2["data_version"] != r1["data_version"]
    assert r2["data_version"].startswith("v2-")
    assert r2["lotes_novos"] == ["lote_02"]
    for nome in PARTICOES:
        assert antes[nome] <= depois[nome], f"registro antigo saiu de {nome}"


def test_lote_alterado_e_rejeitado(cfg_tmp):
    cfg, tmp = cfg_tmp
    executar_etl(cfg)
    with open(tmp / "dir_raw" / "lote_01.csv", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="imutáveis"):
        executar_etl(cfg)


def test_particoes_disjuntas_e_sem_duplicados(cfg_tmp):
    cfg, tmp = cfg_tmp
    shutil.copy(LOTE2, tmp / "dir_raw" / "lote_02.csv")
    executar_etl(cfg)
    e = _enderecos(cfg)
    assert not (e["treino"] & e["validacao"]) and not (e["treino"] & e["teste"]) \
        and not (e["validacao"] & e["teste"])
    for p in carregar_particoes(cfg):
        assert not p[ID].duplicated().any()


def test_preprocessador_ajusta_so_no_treino(cfg_tmp):
    cfg, _ = cfg_tmp
    executar_etl(cfg)
    import json
    schema = json.loads((Path(cfg["dados"]["dir_processed"]) / "schema.json").read_text("utf-8"))
    treino, validacao, _ = carregar_particoes(cfg)
    pre = montar_preprocessador(schema["features"], cfg).fit(treino)
    xv = pre.transform(validacao)
    assert isinstance(xv, pd.DataFrame)
    assert list(xv.columns) == schema["features"]
    assert not xv.isna().any().any()
    assert ALVO not in xv.columns and ID not in xv.columns
