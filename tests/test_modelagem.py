"""Competição A vs B e ciclo de promoção de ponta a ponta (Dev 3).

Dados sintéticos + MLflow SQLite e pastas temporárias: não toca em mlflow.db,
mlartifacts/, artifacts/ nem models/ do repositório.

Rodar a partir da raiz:  python -m pytest -q tests/test_modelagem.py
"""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.ingestao import carregar_config
from src.modelagem import (avaliar, carregar_competicao, carregar_modelo, gravar_competicao,
                           montar_estimador, razao_classes, registrar_candidato,
                           resolver_ciclo, tabela_comparativa, treinar_candidatos)
from src.promocao import (MANTIDO, PROMOVIDO, carregar_historico, carregar_manifesto,
                          executar_promocao, gravar_decisao, registrar_decisao_mlflow)

FEATURES = [f"f{i}" for i in range(1, 9)]
ELEITAS = ["f1", "f2", "f3"]


def _particao(n, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    df["f3"] = np.abs(df["f3"]) * 100          # cauda pesada, como os valores em ETH
    df.loc[rng.random(n) < 0.05, "f2"] = np.nan  # ausentes, como as colunas ERC20
    logit = 2.0 * df["f1"] - 1.5 * df["f2"].fillna(0) + 0.01 * df["f3"] - 1.5
    df["flag"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["address"] = [f"0x{seed:02d}{i:06d}" for i in range(n)]
    df["lote"] = "lote_01"
    return df[["address", "lote", "flag"] + FEATURES]


@pytest.fixture
def dados():
    return _particao(600, 1), _particao(250, 2), _particao(250, 3)


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    import mlflow

    c = copy.deepcopy(carregar_config())
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    c["projeto"]["tracking_uri"] = uri
    c["projeto"]["experimento_mlflow"] = "teste_modelagem"
    c["votacao"]["dir_artifacts"] = str(tmp_path / "artifacts")
    c["dados"]["dir_processed"] = str(tmp_path / "processed")
    c["promocao"]["manifesto"] = str(tmp_path / "models" / "modelo_vigente.json")
    c["promocao"]["historico"] = str(tmp_path / "models" / "historico_decisoes.csv")
    c["modelos"]["hiperparametros"]["xgboost"]["n_estimators"] = 40   # teste rápido
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("CYCLE_ID", raising=False)
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    # experimento criado antes, com artefatos na pasta temporária
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("teste_modelagem", artifact_location=(tmp_path / "mlart").as_uri())
    return c


def _ciclo(cycle_id, dv="v1-aaaa0000"):
    return {"cycle_id": cycle_id, "data_version": dv, "features": FEATURES,
            "eleitas": ELEITAS, "votacao_run_id": "votacao-xyz"}


# --------------------------------------------------------------------------- #
# Montagem e treino
# --------------------------------------------------------------------------- #
def test_scale_pos_weight_vem_do_treino():
    y = pd.Series([0] * 78 + [1] * 22)
    assert razao_classes(y) == pytest.approx(78 / 22, rel=1e-5)
    est, params = montar_estimador("xgboost", y)
    assert params["scale_pos_weight"] == pytest.approx(78 / 22, rel=1e-5)
    assert params["random_state"] == 42
    _, params_lr = montar_estimador("logistic_regression", y)
    assert params_lr["class_weight"] == "balanced" and params_lr["random_state"] == 42
    with pytest.raises(ValueError):
        montar_estimador("svm", y)


def test_os_dois_modelos_usam_as_mesmas_features_eleitas(cfg, dados):
    treino, val, _ = dados
    cand = treinar_candidatos(treino, val, FEATURES, ELEITAS, cfg)
    assert set(cand) == {"A", "B"}
    assert cand["A"]["algoritmo"] == "logistic_regression"
    assert cand["B"]["algoritmo"] == "xgboost"
    for c in cand.values():
        est = c["pipeline"].named_steps["modelo"]
        assert list(est.feature_names_in_) == ELEITAS          # seleção por nome
        m = c["metricas_validacao"]
        assert m["n"] == len(val) and m["tp"] + m["fn"] == int(val["flag"].sum())
        assert 0 <= m["f1"] <= 1
    tab = tabela_comparativa(cand)
    assert list(tab.index) == ["A", "B"] and "f1" in tab.columns


def test_preprocessamento_ajustado_so_no_treino(cfg, dados):
    treino, val, _ = dados
    c = treinar_candidatos(treino, val, FEATURES, ELEITAS, cfg)["A"]
    imputer = c["pipeline"].named_steps["pre"].named_transformers_["num"].named_steps["imputer"]
    log_f2 = np.sign(treino["f2"]) * np.log1p(np.abs(treino["f2"]))
    assert imputer.statistics_[FEATURES.index("f2")] == pytest.approx(log_f2.median())


def test_treino_deterministico(cfg, dados):
    treino, val, _ = dados
    p1 = {n: c["pipeline"].predict_proba(val)[:, 1]
          for n, c in treinar_candidatos(treino, val, FEATURES, ELEITAS, cfg).items()}
    p2 = {n: c["pipeline"].predict_proba(val)[:, 1]
          for n, c in treinar_candidatos(treino, val, FEATURES, ELEITAS, cfg).items()}
    for n in p1:
        np.testing.assert_array_equal(p1[n], p2[n])


# --------------------------------------------------------------------------- #
# Ciclo de ponta a ponta: 03 -> 04, duas vezes
# --------------------------------------------------------------------------- #
def _rodar_ciclo(cfg, dados, cycle_id, dv):
    treino, val, teste = dados
    ciclo = _ciclo(cycle_id, dv)
    cand = treinar_candidatos(treino, val, FEATURES, ELEITAS, cfg)
    for c in cand.values():
        registrar_candidato(c, ciclo, val, cfg)
    gravar_competicao(cand, ciclo, cfg)
    comp = carregar_competicao(cycle_id, cfg)
    res = executar_promocao(comp, val, teste, cfg)
    registrar_decisao_mlflow(res, cfg)
    gravar_decisao(res, cfg)
    return comp, res


def test_ciclo_1_promove_e_ciclo_2_sem_mudanca_mantem(cfg, dados):
    import mlflow

    _, val, teste = dados
    assert carregar_manifesto(cfg) is None

    # ---- ciclo 1: não há vigente -> promove o vencedor --------------------------
    comp1, res1 = _rodar_ciclo(cfg, dados, "c1", "v1-aaaa0000")
    assert res1["decisao"]["decisao"] == PROMOVIDO
    man1 = carregar_manifesto(cfg)
    venc = comp1["candidatos"][res1["vencedor"]]
    assert man1["run_id"] == venc["run_id"] and man1["decisao"] == PROMOVIDO
    assert man1["features"] == ELEITAS and man1["treinado_no_ciclo"] == "c1"
    assert man1["f1_validacao"] == pytest.approx(venc["metricas_validacao"]["f1"])
    assert man1["f1_teste"] is not None and man1["vigente_anterior"] is None
    assert man1["registry"]["alias"] == "vigente"

    # runs do ciclo no MLflow: A, B e a decisão, todos com o mesmo cycle_id
    runs = mlflow.search_runs(experiment_names=["teste_modelagem"],
                              filter_string="tags.cycle_id = 'c1'")
    assert sorted(runs["tags.etapa"]) == ["promocao", "treino_competicao", "treino_competicao"]
    treinos = runs[runs["tags.etapa"] == "treino_competicao"]
    assert set(treinos["tags.algoritmo"]) == {"logistic_regression", "xgboost"}
    assert set(treinos["tags.data_version"]) == {"v1-aaaa0000"}
    assert treinos["metrics.val_f1"].notna().all()

    # o vigente carregado pelo run_id do manifesto reproduz o F1 de validação
    modelo = carregar_modelo(man1["run_id"], cfg)
    assert avaliar(modelo, val, cfg)["f1"] == pytest.approx(man1["f1_validacao"])
    assert modelo.predict_proba(teste)[:, 1].shape == (len(teste),)

    # ---- ciclo 2 com os mesmos dados: F1 igual ao vigente -> mantido ------------
    _, res2 = _rodar_ciclo(cfg, dados, "c2", "v1-aaaa0000")
    assert res2["decisao"]["decisao"] == MANTIDO
    man2 = carregar_manifesto(cfg)
    assert man2["run_id"] == man1["run_id"]                     # ponteiro não mudou
    assert man2["cycle_id"] == "c2" and man2["treinado_no_ciclo"] == "c1"
    assert man2["vigente_anterior"]["run_id"] == man1["run_id"]
    assert man2["registry"] == man1["registry"]
    assert "margem" in man2["motivo"]

    hist = carregar_historico(cfg)
    assert list(hist["cycle_id"]) == ["c1", "c2"]
    assert list(hist["decisao"]) == [PROMOVIDO, MANTIDO]
    # modelos anteriores continuam no MLflow
    assert mlflow.get_run(man1["run_id"]).info.lifecycle_stage == "active"


def test_competicao_json_tem_contrato_para_o_04_e_o_painel(cfg, dados):
    treino, val, _ = dados
    cand = treinar_candidatos(treino, val, FEATURES, ELEITAS, cfg)
    for c in cand.values():
        c["run_id"] = f"run-{c['modelo']}"
    arq = gravar_competicao(cand, _ciclo("c9"), cfg)
    js = json.loads(arq.read_text("utf-8"))
    assert js["cycle_id"] == "c9" and js["features"] == ELEITAS
    for n in ("A", "B"):
        c = js["candidatos"][n]
        assert {"algoritmo", "run_id", "model_uri", "params", "metricas_validacao"} <= set(c)
        assert {"f1", "precision", "recall", "tp", "fp", "fn", "tn"} <= set(c["metricas_validacao"])
    assert (arq.parent / "comparacao_modelos.csv").exists()


# --------------------------------------------------------------------------- #
# resolver_ciclo: contrato com o Dev 2
# --------------------------------------------------------------------------- #
def _gravar_contratos(cfg, tmp_path, dv_schema, dv_votacao, eleitas=ELEITAS, cycle_id="c1"):
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "schema.json").write_text(json.dumps({"data_version": dv_schema, "features": FEATURES}), "utf-8")
    pasta = tmp_path / "artifacts" / cycle_id
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / "features_eleitas.json").write_text(json.dumps(
        {"cycle_id": cycle_id, "data_version": dv_votacao, "features": eleitas,
         "mlflow_run_id": "vot-1"}), "utf-8")


def test_resolver_ciclo(cfg, tmp_path, monkeypatch):
    _gravar_contratos(cfg, tmp_path, "v1-aaaa0000", "v1-aaaa0000")
    c = resolver_ciclo(cfg=cfg)                       # sem CYCLE_ID: o mais recente
    assert c["cycle_id"] == "c1" and c["eleitas"] == ELEITAS and c["features"] == FEATURES
    assert c["votacao_run_id"] == "vot-1"
    monkeypatch.setenv("CYCLE_ID", "outro")
    with pytest.raises(FileNotFoundError):            # com CYCLE_ID: estrito
        resolver_ciclo(cfg=cfg)


def test_resolver_ciclo_recusa_votacao_de_dados_antigos(cfg, tmp_path):
    _gravar_contratos(cfg, tmp_path, "v2-bbbb0000", "v1-aaaa0000")
    with pytest.raises(ValueError, match="data_version"):
        resolver_ciclo("c1", cfg)
    _gravar_contratos(cfg, tmp_path, "v1-aaaa0000", "v1-aaaa0000", eleitas=["f1", "zzz"])
    with pytest.raises(ValueError, match="ausentes"):
        resolver_ciclo("c1", cfg)
