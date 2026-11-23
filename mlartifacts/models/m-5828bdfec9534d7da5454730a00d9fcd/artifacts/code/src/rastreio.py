"""Rastreio do ciclo: ``cycle_id``, MLflow e convenção de tags (Dev 4).

Contrato com as outras frentes
------------------------------
- ``obter_cycle_id()``: o ``cycle_id`` do ciclo. No workflow vem da variável
  ``CYCLE_ID`` (``AAAAMMDD-HHMM-r<run_number>``, ex.: ``20261110-0900-r12``);
  ``run_pipeline.sh`` gera um quando a variável falta. Rodando um notebook
  solto, o fallback é ``AAAAMMDD-HHMM-local``. O sufixo ``-local`` deixa
  ``artifacts/<cycle_id>/`` fora do git.
- ``configurar_mlflow()``: aponta o MLflow para ``sqlite:///<raiz>/mlflow.db``
  e garante o experimento com os artefatos em ``<raiz>/mlartifacts``.
- ``iniciar_run(etapa, ...)``: abre um run já com as tags obrigatórias
  ``cycle_id``, ``data_version`` e ``etapa``.
- ``relocalizar_artefatos()``: o MLflow grava caminhos ABSOLUTOS dos artefatos
  no ``mlflow.db``. Como o banco é commitado pelo workflow (caminho do runner)
  e aberto em outras máquinas, os caminhos são reescritos para a raiz atual
  antes de cada ciclo e antes de abrir a UI.

Etapas (tag ``etapa``): ``etl`` (01), ``votacao_features`` (02),
``treino_competicao`` (03), ``promocao`` (04) e ``previsoes`` (05).

Linha de comando (a partir da raiz):
    python -m src.rastreio cycle-id      # imprime/gera o cycle_id
    python -m src.rastreio preparar      # experimento + caminhos do mlflow.db
    python -m src.rastreio ui            # relocaliza e abre o mlflow ui
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.ingestao import RAIZ, carregar_config

ETAPAS = ("etl", "votacao_features", "treino_competicao", "promocao", "previsoes")
TAGS_OBRIGATORIAS = ("cycle_id", "data_version", "etapa")
DIR_ARTEFATOS_MLFLOW = "mlartifacts"
DB_PADRAO = "mlflow.db"


# --------------------------------------------------------------------------- #
# cycle_id
# --------------------------------------------------------------------------- #
def gerar_cycle_id(run_number: str | int | None = None, agora: datetime | None = None) -> str:
    """``AAAAMMDD-HHMM-r<run_number>`` (UTC); sem ``run_number``: ``...-local``."""
    agora = agora or datetime.now(timezone.utc)
    base = agora.strftime("%Y%m%d-%H%M")
    return f"{base}-r{run_number}" if run_number not in (None, "") else f"{base}-local"


def obter_cycle_id() -> str:
    """``CYCLE_ID`` do ambiente (workflow ou ``run_pipeline.sh``); senão fallback local.

    O fallback usa a hora local, como os notebooks 01 e 02 já faziam, para
    que o ``cycle_id`` de uma execução manual seja fácil de reconhecer.
    """
    cid = os.environ.get("CYCLE_ID", "").strip()
    if cid:
        return cid
    return datetime.now().strftime("%Y%m%d-%H%M") + "-local"


def eh_local(cycle_id: str) -> bool:
    return cycle_id.endswith("-local")


# --------------------------------------------------------------------------- #
# data_version
# --------------------------------------------------------------------------- #
def obter_data_version(cfg: dict | None = None) -> str | None:
    """``data_version`` das partições atuais (``schema.json``) ou do ``lotes.json``."""
    cfg = cfg or carregar_config()
    proc = RAIZ / cfg["dados"]["dir_processed"]
    for nome in ("schema.json", "lotes.json"):
        p = proc / nome
        if p.exists():
            dv = json.loads(p.read_text(encoding="utf-8")).get("data_version")
            if dv:
                return dv
    return None


# --------------------------------------------------------------------------- #
# MLflow
# --------------------------------------------------------------------------- #
def caminho_db(cfg: dict | None = None) -> Path:
    """Arquivo SQLite do tracking (``config.yaml`` -> ``projeto.tracking_uri``)."""
    cfg = cfg or carregar_config()
    uri = str(cfg["projeto"]["tracking_uri"])
    prefixo = "sqlite:///"
    rel = uri[len(prefixo):] if uri.startswith(prefixo) else DB_PADRAO
    p = Path(rel)
    return p if p.is_absolute() else (RAIZ / p).resolve()


def tracking_uri(cfg: dict | None = None) -> str:
    """``MLFLOW_TRACKING_URI`` se definida; senão o SQLite do config, absoluto.

    Absoluto para dar no mesmo ``mlflow.db`` rodando de ``notebooks/`` ou da raiz.
    """
    env = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if env:
        return env
    return "sqlite:///" + caminho_db(cfg).as_posix()


def dir_artefatos_mlflow(cfg: dict | None = None) -> Path:
    """Pasta de artefatos: ``mlartifacts/`` ao lado do ``mlflow.db`` em uso.

    Para o banco do projeto é ``<raiz>/mlartifacts``; um banco temporário
    (testes) ganha a sua própria pasta e nunca suja a do repositório.
    """
    uri = tracking_uri(cfg)
    if uri.startswith("sqlite:///"):
        return (Path(uri[len("sqlite:///"):]).parent / DIR_ARTEFATOS_MLFLOW).resolve()
    return (RAIZ / DIR_ARTEFATOS_MLFLOW).resolve()


def configurar_mlflow(cfg: dict | None = None):
    """Aponta o MLflow para o tracking do projeto e ativa o experimento.

    O experimento é criado com ``artifact_location = <pasta do db>/mlartifacts``
    (``<raiz>/mlartifacts``, commitada pelo workflow), nunca no ``./mlruns`` padrão.
    """
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
    import mlflow

    cfg = cfg or carregar_config()
    mlflow.set_tracking_uri(tracking_uri(cfg))
    nome = cfg["projeto"]["experimento_mlflow"]
    if mlflow.get_experiment_by_name(nome) is None:
        mlflow.create_experiment(nome, artifact_location=dir_artefatos_mlflow(cfg).as_uri())
    mlflow.set_experiment(nome)
    return mlflow


def tags_do_ciclo(etapa: str, cycle_id: str | None = None, data_version: str | None = None,
                  cfg: dict | None = None, **extras) -> dict:
    """Tags padrão de qualquer run do pipeline."""
    if etapa not in ETAPAS:
        raise ValueError(f"etapa desconhecida: {etapa!r}; válidas: {list(ETAPAS)}")
    dv = data_version or obter_data_version(cfg)
    if not dv:
        raise ValueError("data_version indisponível: rode o notebook 01 antes")
    tags = {"cycle_id": cycle_id or obter_cycle_id(), "data_version": dv, "etapa": etapa}
    gatilho = os.environ.get("GITHUB_EVENT_NAME")
    if gatilho:
        tags["gatilho"] = gatilho
    if os.environ.get("GITHUB_RUN_ID"):
        tags["github_run_id"] = os.environ["GITHUB_RUN_ID"]
    tags.update({k: str(v) for k, v in extras.items() if v is not None})
    return tags


@contextmanager
def iniciar_run(etapa: str, run_name: str | None = None, cycle_id: str | None = None,
                data_version: str | None = None, cfg: dict | None = None, **tags_extras):
    """``with iniciar_run("previsoes") as run:`` -- run já com as tags do ciclo."""
    mlflow = configurar_mlflow(cfg)
    tags = tags_do_ciclo(etapa, cycle_id, data_version, cfg, **tags_extras)
    with mlflow.start_run(run_name=run_name or f"{etapa}_{tags['cycle_id']}", tags=tags) as run:
        yield run


def runs_do_ciclo(cycle_id: str, cfg: dict | None = None):
    """DataFrame com os runs de um ciclo (todas as etapas)."""
    mlflow = configurar_mlflow(cfg)
    cfg = cfg or carregar_config()
    return mlflow.search_runs(
        experiment_names=[cfg["projeto"]["experimento_mlflow"]],
        filter_string=f"tags.cycle_id = '{cycle_id}'",
        order_by=["attributes.start_time ASC"],
    )


# --------------------------------------------------------------------------- #
# Caminhos absolutos dentro do mlflow.db
# --------------------------------------------------------------------------- #
def _raizes_antigas(con: sqlite3.Connection) -> set[str]:
    """Raízes de repositório gravadas no banco que não são a atual.

    A raiz é deduzida do ``artifact_location`` dos experimentos que terminam
    em ``/mlartifacts`` (o do projeto): tudo antes disso é a raiz antiga. A
    raiz atual é a pasta onde o banco está agora.
    """
    atual = Path(con.execute("PRAGMA database_list").fetchone()[2]).resolve().parent.as_posix()
    antigas = set()
    for (loc,) in con.execute("SELECT artifact_location FROM experiments"):
        if not loc:
            continue
        caminho = loc[len("file://"):] if loc.startswith("file://") else loc
        caminho = caminho.rstrip("/")
        sufixo = "/" + DIR_ARTEFATOS_MLFLOW
        if caminho.endswith(sufixo):
            raiz = caminho[: -len(sufixo)]
            if raiz and raiz != atual:
                antigas.add(raiz)
    return antigas


def relocalizar_artefatos(cfg: dict | None = None, db: Path | None = None) -> dict:
    """Reescreve, no ``mlflow.db``, os caminhos absolutos para a pasta atual do banco.

    Troca a raiz antiga (ex.: ``/home/runner/work/ml_ops_eth/ml_ops_eth``) pela
    atual em todas as colunas de texto que a contêm: experimentos, runs,
    modelos logados (``logged_models``) e versões do registry. Não faz nada se
    o banco não existe ou se os caminhos já estão certos.
    Devolve ``{"antigas": [...], "linhas": n}``.
    """
    db = Path(db) if db else caminho_db(cfg)
    if not db.exists():
        return {"antigas": [], "linhas": 0}
    atual = db.resolve().parent.as_posix()   # a raiz do repositório, para o banco do projeto
    linhas = 0
    with sqlite3.connect(db) as con:
        antigas = _raizes_antigas(con)
        if not antigas:
            return {"antigas": [], "linhas": 0}
        tabelas = [t for (t,) in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
        for tabela in tabelas:
            colunas = [c[1] for c in con.execute(f'PRAGMA table_info("{tabela}")')
                       if (c[2] or "").upper().startswith(("VARCHAR", "TEXT", "CHAR"))]
            for col in colunas:
                for antiga in antigas:
                    # com a barra final: nunca casa um prefixo parcial de outro diretório
                    cur = con.execute(
                        f'UPDATE "{tabela}" SET "{col}" = REPLACE("{col}", ?, ?) '
                        f'WHERE instr("{col}", ?) > 0',
                        (antiga + "/", atual + "/", antiga + "/"),
                    )
                    linhas += cur.rowcount
        con.commit()
    return {"antigas": sorted(antigas), "linhas": linhas}


def preparar(cfg: dict | None = None) -> dict:
    """Passo inicial de cada ciclo: relocaliza caminhos e garante o experimento."""
    cfg = cfg or carregar_config()
    reloc = relocalizar_artefatos(cfg)
    mlflow = configurar_mlflow(cfg)
    exp = mlflow.get_experiment_by_name(cfg["projeto"]["experimento_mlflow"])
    return {"tracking_uri": tracking_uri(cfg), "experimento": exp.name,
            "experiment_id": exp.experiment_id, "artifact_location": exp.artifact_location,
            "relocalizacao": reloc}


# --------------------------------------------------------------------------- #
def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("cycle-id", help="imprime o CYCLE_ID do ambiente ou gera um novo")
    g.add_argument("--run-number", default=os.environ.get("GITHUB_RUN_NUMBER"))
    sub.add_parser("preparar", help="relocaliza caminhos do mlflow.db e garante o experimento")
    u = sub.add_parser("ui", help="relocaliza e abre o mlflow ui")
    u.add_argument("--port", default="5000")
    args = ap.parse_args()

    if args.cmd == "cycle-id":
        print(os.environ.get("CYCLE_ID", "").strip() or gerar_cycle_id(args.run_number))
    elif args.cmd == "preparar":
        print(json.dumps(preparar(), indent=2, ensure_ascii=False))
    else:
        relocalizar_artefatos()
        os.execvp("mlflow", ["mlflow", "ui", "--backend-store-uri", tracking_uri(),
                             "--port", str(args.port)])


if __name__ == "__main__":
    _cli()
