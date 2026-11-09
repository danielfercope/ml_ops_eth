"""Limpeza, partição determinística e pré-processamento.

Contrato para as outras frentes
-------------------------------
- ``data/processed/{treino,validacao,teste}.parquet``: colunas ``address``,
  ``flag``, ``lote`` + as features limpas (nomes em snake_case).
- ``data/processed/schema.json``: lista oficial de ``features``, o alvo,
  o data_version e as colunas descartadas (e o motivo).
- ``montar_preprocessador(features)``: ColumnTransformer NÃO ajustado
  (log com sinal -> imputer mediana -> StandardScaler). Deve ser ajustado só no
  treino, dentro do Pipeline do modelo (Dev 3). Sai DataFrame com os mesmos
  nomes de coluna, então dá para selecionar as features eleitas por nome.

Uso:
    from src.preprocessamento import carregar_particoes, carregar_schema, montar_preprocessador
    treino, validacao, teste = carregar_particoes()
    features = carregar_schema()["features"]
"""
from __future__ import annotations

import hashlib
import json
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.ingestao import RAIZ, _caminho, carregar_config, ingerir

PARTICOES = ("treino", "validacao", "teste")
ID, ALVO, LOTE = "address", "flag", "lote"


# --------------------------------------------------------------------------- #
# Limpeza
# --------------------------------------------------------------------------- #
def normalizar_nome(coluna: str) -> str:
    """' ERC20 total Ether received' -> 'erc20_total_ether_received'."""
    nome = re.sub(r"[^0-9a-zA-Z]+", "_", coluna.strip()).strip("_").lower()
    return nome


def limpar(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Remove colunas que nunca são features e padroniza os nomes.

    Operações linha a linha / por coluna fixa: não dependem da partição,
    portanto não há vazamento.
    """
    d = cfg["dados"]
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    descartadas = {c: "identificador/índice" for c in d["colunas_descartadas"]}
    descartadas |= {c: "categórica de alta cardinalidade" for c in d["colunas_categoricas_descartadas"]}

    df = df.drop(columns=[c for c in descartadas if c in df.columns and c != d["id"]])
    df = df.rename(columns={d["id"]: ID, d["alvo"]: ALVO, "_lote": LOTE})
    df.columns = [c if c in (ID, ALVO, LOTE) else normalizar_nome(c) for c in df.columns]
    df[ID] = df[ID].str.strip().str.lower()
    df[ALVO] = df[ALVO].astype(int)
    return df, descartadas


# --------------------------------------------------------------------------- #
# Partição determinística por hash do Address
# --------------------------------------------------------------------------- #
def particao_do_endereco(address: str, sal: str, fracoes: tuple[float, float]) -> str:
    """Mesmo endereço -> sempre a mesma partição, em qualquer ciclo."""
    h = int(hashlib.md5(f"{sal}:{address}".encode()).hexdigest()[:8], 16)
    u = h / 0xFFFFFFFF
    if u < fracoes[0]:
        return "treino"
    if u < fracoes[0] + fracoes[1]:
        return "validacao"
    return "teste"


def particionar(df: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    p = cfg["particao"]
    fr = (p["treino"], p["validacao"])
    sal = str(cfg["seeds"]["particao"])
    rotulo = df[ID].map(lambda a: particao_do_endereco(a, sal, fr))
    return {nome: df.loc[rotulo == nome].reset_index(drop=True) for nome in PARTICOES}


# --------------------------------------------------------------------------- #
# Pré-processamento (ajustado SÓ no treino, dentro do Pipeline do modelo)
# --------------------------------------------------------------------------- #
def log_com_sinal(x):
    """sign(x) * log1p(|x|): comprime caudas pesadas, aceita negativos, sem estado."""
    return np.sign(x) * np.log1p(np.abs(x))


def montar_preprocessador(features: list[str], cfg: dict | None = None) -> ColumnTransformer:
    cfg = cfg or carregar_config()
    pp = cfg["preprocessamento"]
    passos = []
    if pp.get("transformacao_log", True):
        passos.append(("log", FunctionTransformer(log_com_sinal, feature_names_out="one-to-one")))
    passos += [
        ("imputer", SimpleImputer(strategy=pp.get("imputer", "median"))),
        ("scaler", StandardScaler()),
    ]
    ct = ColumnTransformer(
        [("num", Pipeline(passos), list(features))],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return ct.set_output(transform="pandas")


def colunas_constantes(treino: pd.DataFrame, candidatas: list[str]) -> list[str]:
    """Colunas com no máximo 1 valor distinto NO TREINO (sem olhar val/teste)."""
    return [c for c in candidatas if treino[c].nunique(dropna=True) <= 1]


# --------------------------------------------------------------------------- #
# ETL completo
# --------------------------------------------------------------------------- #
def executar_etl(cfg: dict | None = None, cycle_id: str | None = None) -> dict:
    """Ingestão -> limpeza -> partição -> features -> parquets + schema.json."""
    cfg = cfg or carregar_config()
    bruto, meta = ingerir(cfg, cycle_id=cycle_id)
    df, descartadas = limpar(bruto, cfg)
    partes = particionar(df, cfg)

    candidatas = [c for c in df.columns if c not in (ID, ALVO, LOTE)]
    constantes = colunas_constantes(partes["treino"], candidatas)
    features = [c for c in candidatas if c not in constantes]
    descartadas |= {c: "constante no treino" for c in constantes}

    out = _caminho(cfg, "dir_processed")
    out.mkdir(parents=True, exist_ok=True)
    resumo_part = {}
    for nome, parte in partes.items():
        parte = parte[[ID, LOTE, ALVO] + features]
        parte.to_parquet(out / f"{nome}.parquet", index=False)
        resumo_part[nome] = {
            "linhas": int(len(parte)),
            "fraudes": int(parte[ALVO].sum()),
            "taxa_fraude": round(float(parte[ALVO].mean()), 4),
        }

    schema = {
        "data_version": meta["data_version"],
        "id": ID,
        "alvo": ALVO,
        "features": features,
        "n_features": len(features),
        "descartadas": descartadas,
        "particoes": {n: f"{cfg['dados']['dir_processed']}/{n}.parquet" for n in PARTICOES},
        "resumo_particoes": resumo_part,
    }
    (out / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
    return {**meta, "n_features": len(features), "constantes": constantes,
            "particoes": resumo_part}


def carregar_schema(cfg: dict | None = None) -> dict:
    cfg = cfg or carregar_config()
    return json.loads((_caminho(cfg, "dir_processed") / "schema.json").read_text(encoding="utf-8"))


def carregar_particoes(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or carregar_config()
    out = _caminho(cfg, "dir_processed")
    return tuple(pd.read_parquet(out / f"{n}.parquet") for n in PARTICOES)  # type: ignore[return-value]


if __name__ == "__main__":
    print(json.dumps(executar_etl(), indent=2, ensure_ascii=False))
