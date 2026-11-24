"""Ingestão de dados: lotes, deduplicação e data_version.

Fluxo:
    1. ``gerar_lotes``  — (uma vez) divide o CSV original do Kaggle em
       lote_01 (~70%) e lote_02 (~30%), estratificado por FLAG e com seed fixa.
       lote_01 vai para ``data/raw``; lote_02 fica em ``data/reserva`` até o ciclo 2.
    2. ``ingerir``      — lê todos os lotes presentes em ``data/raw``, identifica
       os novos (ainda não registrados em ``lotes.json``), remove duplicados por
       Address (a primeira ocorrência vence, então um lote novo nunca sobrescreve
       um registro antigo) e calcula o ``data_version``.

Uso pela linha de comando (a partir da raiz do repositório):
    python -m src.ingestao gerar-lotes
    python -m src.ingestao ingerir
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parents[1]
CONFIG_PADRAO = RAIZ / "config" / "config.yaml"


# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #
def carregar_config(caminho: str | Path = CONFIG_PADRAO) -> dict:
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _caminho(cfg: dict, chave: str) -> Path:
    return RAIZ / cfg["dados"][chave]


def _relativo(p: Path) -> str:
    try:
        return p.resolve().relative_to(RAIZ).as_posix()
    except ValueError:
        return p.as_posix()


def _sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 16), b""):
            h.update(bloco)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. Download e geração dos lotes
# --------------------------------------------------------------------------- #
def baixar_original(cfg: dict | None = None) -> Path:
    """Garante o CSV original em data/original (baixa com kagglehub se faltar).

    O dataset é público (licença ODbL), então funciona sem credenciais; se o
    Kaggle exigir login, defina KAGGLE_USERNAME e KAGGLE_KEY.
    """
    cfg = cfg or carregar_config()
    destino_dir = _caminho(cfg, "dir_original")
    destino = destino_dir / cfg["dados"]["arquivo_original"]
    if destino.exists():
        return destino
    import kagglehub  # import tardio: só é necessário para gerar os lotes

    origem = Path(kagglehub.dataset_download(cfg["dados"]["kaggle_dataset"]))
    destino_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(origem / cfg["dados"]["arquivo_original"], destino)
    return destino


def gerar_lotes(cfg: dict | None = None, sobrescrever: bool = False) -> dict[str, Path]:
    """Divide o CSV original em lote_01 / lote_02 (estratificado, seed fixa).

    Determinístico: rodar de novo gera arquivos idênticos byte a byte.
    """
    cfg = cfg or carregar_config()
    df = pd.read_csv(baixar_original(cfg))
    alvo = cfg["dados"]["alvo"]
    lotes = cfg["dados"]["lotes"]
    assert len(lotes) == 2, "a divisão atual prevê exatamente 2 lotes"

    l1, l2 = train_test_split(
        df,
        train_size=lotes[0]["fracao"],
        stratify=df[alvo],
        random_state=cfg["seeds"]["lotes"],
    )
    # mantém a ordem original dentro de cada lote (diffs legíveis no git)
    l1, l2 = l1.sort_index(), l2.sort_index()

    destinos = {
        lotes[0]["nome"]: _caminho(cfg, "dir_raw") / f"{lotes[0]['nome']}.csv",
        lotes[1]["nome"]: _caminho(cfg, "dir_reserva") / f"{lotes[1]['nome']}.csv",
    }
    # se o lote_02 já foi movido para data/raw (ciclo 2), não recria na reserva
    ja_em_raw = _caminho(cfg, "dir_raw") / f"{lotes[1]['nome']}.csv"
    for (nome, destino), dados in zip(destinos.items(), (l1, l2)):
        if nome == lotes[1]["nome"] and ja_em_raw.exists():
            destinos[nome] = ja_em_raw
            continue
        if destino.exists() and not sobrescrever:
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        dados.to_csv(destino, index=False, lineterminator="\n")
    return destinos


# --------------------------------------------------------------------------- #
# 2. Ingestão incremental
# --------------------------------------------------------------------------- #
def _registro_path(cfg: dict) -> Path:
    return _caminho(cfg, "dir_processed") / "lotes.json"


def ler_registro(cfg: dict | None = None) -> dict:
    cfg = cfg or carregar_config()
    p = _registro_path(cfg)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"lotes": [], "data_version": None}


def calcular_data_version(hashes_lotes: list[str]) -> str:
    """``v<n>-<hash8>``: n = nº de lotes; hash8 = sha256 dos hashes em ordem."""
    combinado = hashlib.sha256("|".join(hashes_lotes).encode()).hexdigest()[:8]
    return f"v{len(hashes_lotes)}-{combinado}"


def ingerir(cfg: dict | None = None, cycle_id: str | None = None,
            salvar_registro: bool = True) -> tuple[pd.DataFrame, dict]:
    """Lê os lotes de data/raw e devolve o dataset consolidado + metadados.

    - Lotes são processados em ordem de nome (lote_01, lote_02, ...).
    - Um lote já registrado cujo conteúdo mudou é um erro (lotes são imutáveis).
    - Duplicados por Address: fica a primeira ocorrência, então registros
      antigos nunca são substituídos por um lote novo.
    """
    cfg = cfg or carregar_config()
    id_col = cfg["dados"]["id"]
    registro = ler_registro(cfg)
    conhecidos = {l["nome"]: l for l in registro["lotes"]}

    arquivos = sorted(_caminho(cfg, "dir_raw").glob(cfg["dados"]["padrao_lotes"]))
    if not arquivos:
        raise FileNotFoundError(
            "Nenhum lote em data/raw. Rode `python -m src.ingestao gerar-lotes`."
        )

    partes, info_lotes, novos = [], [], []
    for arq in arquivos:
        nome, sha = arq.stem, _sha256_arquivo(arq)
        if nome in conhecidos and conhecidos[nome]["sha256"] != sha:
            raise ValueError(
                f"{arq.name} mudou desde que foi ingerido. Lotes são imutáveis; "
                "crie um lote novo em vez de editar um antigo."
            )
        df_lote = pd.read_csv(arq)
        df_lote["_lote"] = nome
        partes.append(df_lote)
        if nome in conhecidos:
            info = conhecidos[nome]
        else:
            novos.append(nome)
            info = {
                "nome": nome,
                "arquivo": _relativo(arq),
                "sha256": sha,
                "linhas": int(len(df_lote)),
                "fraudes": int((df_lote[cfg["dados"]["alvo"]] == cfg["dados"]["classe_positiva"]).sum()),
                "ingerido_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "cycle_id": cycle_id,
            }
        info_lotes.append(info)

    bruto = pd.concat(partes, ignore_index=True)
    bruto["_address_norm"] = bruto[id_col].str.strip().str.lower()
    duplicados = bruto["_address_norm"].duplicated(keep="first")
    dados = bruto.loc[~duplicados].drop(columns="_address_norm").reset_index(drop=True)

    data_version = calcular_data_version([l["sha256"] for l in info_lotes])
    meta = {
        "data_version": data_version,
        "lotes": [l["nome"] for l in info_lotes],
        "lotes_novos": novos,
        "linhas_brutas": int(len(bruto)),
        "duplicados_removidos": int(duplicados.sum()),
        "linhas": int(len(dados)),
    }

    if salvar_registro:
        _registro_path(cfg).parent.mkdir(parents=True, exist_ok=True)
        _registro_path(cfg).write_text(
            json.dumps({"data_version": data_version, "lotes": info_lotes},
                       indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return dados, meta


# --------------------------------------------------------------------------- #
def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gerar-lotes", help="divide o CSV original em lotes")
    g.add_argument("--sobrescrever", action="store_true")
    sub.add_parser("ingerir", help="consolida os lotes de data/raw")
    args = ap.parse_args()

    if args.cmd == "gerar-lotes":
        for nome, p in gerar_lotes(sobrescrever=args.sobrescrever).items():
            print(f"{nome}: {_relativo(p)}")
    else:
        _, meta = ingerir()
        print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
