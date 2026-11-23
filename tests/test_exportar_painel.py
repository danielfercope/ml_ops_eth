"""Exportador do painel (Dev 5, critério 10).

1. O estado commitado do repositório exporta JSONs que passam no contrato.
2. Um 2º ciclo sintético ("mantido", com vigente reavaliado) também passa,
   cobrindo o caminho que o ciclo 1 sozinho não exercita.
3. Se o contrato quebra, nada é gravado.

Rodar a partir da raiz:  python -m pytest -q tests/test_exportar_painel.py
"""
import copy
import json
import shutil

import pandas as pd
import pytest

from src.contrato_painel import validar_pasta
from src.exportar_painel import Fontes, exportar, montar
from src.ingestao import RAIZ

HIST = RAIZ / "models" / "historico_decisoes.csv"
pytestmark = pytest.mark.skipif(not HIST.exists(), reason="nenhum ciclo decidido ainda")


def _c1() -> str:
    return pd.read_csv(HIST, dtype=str)["cycle_id"].iloc[0]


def test_estado_do_repositorio_exporta_valido(tmp_path):
    pasta = exportar(tmp_path / "data")
    assert validar_pasta(pasta) == []
    d = json.loads((pasta / "ciclos.json").read_text(encoding="utf-8"))
    assert d["origem"] == "pipeline" and d["ciclos"][0]["cycle_id"] == _c1()


@pytest.fixture
def repo_dois_ciclos(tmp_path):
    """Cópia do repositório com um 2º ciclo em que o vigente é mantido."""
    for rel in ("models", "artifacts", "evidencias", "data/raw", "data/processed/lotes.json", "mlflow.db"):
        src, dst = RAIZ / rel, tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, dst)

    c1 = _c1()
    c2 = "20261128-1200-r2"
    a1, a2 = tmp_path / "artifacts" / c1, tmp_path / "artifacts" / c2
    shutil.copytree(a1, a2)
    shutil.copytree(tmp_path / "evidencias" / c1, tmp_path / "evidencias" / c2)

    dec1 = json.loads((a1 / "decisao.json").read_text(encoding="utf-8"))
    comp = json.loads((a2 / "competicao.json").read_text(encoding="utf-8"))
    comp["cycle_id"] = c2
    novos = {"A": "a" * 32, "B": "b" * 32}
    for p, c in comp["candidatos"].items():
        c["run_id"] = novos[p]
    (a2 / "competicao.json").write_text(json.dumps(comp), encoding="utf-8")

    # desafiante com o mesmo F1 do vigente: ganho 0 < margem → mantido
    dec2 = copy.deepcopy(dec1)
    venc = comp["candidatos"]["B"]
    dec2.update({
        "cycle_id": c2, "decisao": "mantido", "decidido_em": "2026-11-28T12:05:00+00:00",
        "motivo": "ganho 0 < margem 0,01",
        "desafiante": {"modelo": "B", "algoritmo": venc["algoritmo"], "run_id": venc["run_id"],
                       "f1_validacao": venc["metricas_validacao"]["f1"]},
        "vigente_anterior": {"run_id": dec1["run_id"], "modelo": dec1["modelo"],
                             "algoritmo": dec1["algoritmo"], "treinado_no_ciclo": dec1["treinado_no_ciclo"],
                             "f1_validacao": dec1["f1_validacao"],
                             "metricas_validacao": dec1["metricas_validacao"]},
    })
    (a2 / "decisao.json").write_text(json.dumps(dec2), encoding="utf-8")
    (tmp_path / "models" / "modelo_vigente.json").write_text(json.dumps(dec2), encoding="utf-8")
    h = pd.read_csv(tmp_path / "models" / "historico_decisoes.csv", dtype=str)
    linha = h.iloc[[0]].copy()
    linha["cycle_id"], linha["decisao"] = c2, "mantido"
    pd.concat([h, linha]).to_csv(tmp_path / "models" / "historico_decisoes.csv", index=False)

    f = Fontes(raiz=tmp_path)
    etl_c1 = f.run_etl(c1)
    etl_c2 = copy.deepcopy(etl_c1)          # mesmo ETL, sem dados novos: o notebook 01
    etl_c2["params"]["lotes_novos"] = "nenhum"   # grava lotes_novos="nenhum"
    f.run_etl = lambda cid: etl_c1 if cid == c1 else etl_c2
    return f, c1, c2


def test_segundo_ciclo_mantido_passa_no_contrato(repo_dois_ciclos, tmp_path):
    f, c1, c2 = repo_dois_ciclos
    pasta = exportar(tmp_path / "saida", fontes=f)
    assert validar_pasta(pasta) == []

    d = montar(f)
    ult = d["ciclos"]["ciclos"][-1]
    assert ult["decisao"] == "mantido" and ult["vigente_anterior"]["cycle_id"] == c1
    assert ult["lotes_novos"] == []
    assert [m["papel"] for m in d["competicao"]["ciclos"][-1]["modelos"]] == ["A", "B", "vigente"]
    assert d["vigente"]["modelo_mudou"] is False and d["vigente"]["anterior"] is None
    assert d["vigente"]["vigente"]["cycle_id"] == c1


def test_contrato_quebrado_nao_grava(repo_dois_ciclos, tmp_path):
    f, _, c2 = repo_dois_ciclos
    p = f.artifacts / c2 / "decisao.json"
    dec = json.loads(p.read_text(encoding="utf-8"))
    dec["decisao"] = "promovido"           # contradiz a regra (ganho 0)
    p.write_text(json.dumps(dec), encoding="utf-8")
    saida = tmp_path / "saida"
    with pytest.raises(ValueError, match="contrato"):
        exportar(saida, fontes=f)
    assert not saida.exists()
