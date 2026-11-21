"""Seleção do vencedor e decisão de promoção (Dev 3).

Fluxo do notebook 04, a cada ciclo:

    1. ``escolher_vencedor``  -- entre A e B, maior F1 de VALIDAÇÃO
       (empate -> ``promocao.desempate_vencedor`` do config, o modelo A).
    2. ``carregar_manifesto`` -- lê ``models/modelo_vigente.json``; o vigente é
       carregado do MLflow pelo ``run_id`` e reavaliado na MESMA validação do
       ciclo atual (os dados mudam entre ciclos, a régua tem de ser a mesma).
    3. ``decidir``            -- aplica a regra fixada no config:
           promove  se  F1_val >= F1_vigente + margem_minima  e  F1_val >= f1_minimo
           sem vigente (ciclo 1): promove se F1_val >= f1_minimo
       senão o vigente é mantido.
    4. Só DEPOIS da decisão o vigente resultante é medido no teste (número do
       relatório); o teste nunca participa da escolha.
    5. ``gravar_decisao``     -- escreve o manifesto e o histórico. É o último
       passo do notebook: se algo falhar antes, nada muda no disco.

Modelos anteriores nunca são apagados: os runs ficam no MLflow e o manifesto
só troca o ponteiro (``run_id``).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features import configurar_mlflow, dir_artifacts
from src.ingestao import RAIZ, carregar_config
from src.modelagem import avaliar, carregar_modelo, model_uri

ETAPA = "promocao"
PROMOVIDO, MANTIDO = "promovido", "mantido"
ARQ_DECISAO = "decisao.json"
ALIAS_VIGENTE = "vigente"
EPS = 1e-9   # tolerância de ponto flutuante na comparação com o limiar

COLUNAS_HISTORICO = [
    "cycle_id", "data_version", "decidido_em",
    "f1_val_A", "f1_val_B",
    "vencedor_modelo", "vencedor_algoritmo", "vencedor_run_id", "f1_val_vencedor",
    "vigente_anterior_run_id", "vigente_anterior_algoritmo", "f1_val_vigente_anterior",
    "f1_necessario", "decisao", "motivo",
    "vigente_run_id", "vigente_modelo", "vigente_algoritmo", "vigente_treinado_no_ciclo",
    "f1_val_vigente", "f1_teste_vigente", "decisao_run_id",
]


# --------------------------------------------------------------------------- #
# Regra (fixada no config.yaml antes do ciclo 1)
# --------------------------------------------------------------------------- #
def carregar_regra_promocao(cfg: dict | None = None) -> dict:
    cfg = cfg or carregar_config()
    p = cfg.get("promocao")
    if not isinstance(p, dict):
        raise ValueError("config.yaml sem a seção 'promocao'")
    for campo in ("margem_minima", "f1_minimo"):
        if campo not in p:
            raise ValueError(f"seção 'promocao' sem o campo '{campo}'")
    regra = {
        "criterio_vencedor": str(p.get("criterio_vencedor", "f1_validacao")),
        "desempate_vencedor": str(p.get("desempate_vencedor", "A")),
        "margem_minima": float(p["margem_minima"]),
        "f1_minimo": float(p["f1_minimo"]),
    }
    if regra["criterio_vencedor"] != "f1_validacao":
        raise ValueError("único critério suportado: f1_validacao")
    if not 0 <= regra["f1_minimo"] <= 1 or regra["margem_minima"] < 0:
        raise ValueError("f1_minimo deve estar em [0, 1] e margem_minima >= 0")
    return regra


def _caminho_cfg(cfg: dict, chave: str, padrao: str) -> Path:
    return RAIZ / cfg.get("promocao", {}).get(chave, padrao)


def caminho_manifesto(cfg: dict | None = None) -> Path:
    return _caminho_cfg(cfg or carregar_config(), "manifesto", "models/modelo_vigente.json")


def caminho_historico(cfg: dict | None = None) -> Path:
    return _caminho_cfg(cfg or carregar_config(), "historico", "models/historico_decisoes.csv")


# --------------------------------------------------------------------------- #
# 1. Vencedor do ciclo
# --------------------------------------------------------------------------- #
def escolher_vencedor(candidatos: dict[str, dict], regra: dict | None = None) -> str:
    """Nome ('A' ou 'B') do candidato com maior F1 de validação.

    ``candidatos`` no formato do ``competicao.json``:
    ``{"A": {"metricas_validacao": {"f1": ...}, ...}, "B": {...}}``.
    Empate (diferença < 1e-9) -> ``regra['desempate_vencedor']``.
    """
    regra = regra or carregar_regra_promocao()
    if not candidatos:
        raise ValueError("nenhum candidato")
    f1 = {n: float(c["metricas_validacao"]["f1"]) for n, c in candidatos.items()}
    melhor = max(f1.values())
    empatados = sorted(n for n, v in f1.items() if melhor - v < EPS)
    preferido = regra.get("desempate_vencedor")
    return preferido if preferido in empatados else empatados[0]


# --------------------------------------------------------------------------- #
# 2. Decisão
# --------------------------------------------------------------------------- #
def decidir(f1_desafiante: float, f1_vigente: float | None, regra: dict | None = None) -> dict:
    """Aplica a regra de promoção e devolve ``{decisao, motivo, f1_necessario}``."""
    regra = regra or carregar_regra_promocao()
    minimo, margem = regra["f1_minimo"], regra["margem_minima"]
    atinge_minimo = f1_desafiante + EPS >= minimo

    if f1_vigente is None:
        if atinge_minimo:
            return {"decisao": PROMOVIDO, "f1_necessario": minimo,
                    "motivo": (f"não há modelo vigente; o vencedor tem F1_val={f1_desafiante:.4f} "
                               f">= mínimo {minimo:.2f}")}
        return {"decisao": MANTIDO, "f1_necessario": minimo,
                "motivo": (f"não há modelo vigente e o vencedor tem F1_val={f1_desafiante:.4f} "
                           f"< mínimo {minimo:.2f}: nada foi promovido")}

    necessario = max(f1_vigente + margem, minimo)
    supera_margem = f1_desafiante + EPS >= f1_vigente + margem
    if supera_margem and atinge_minimo:
        return {"decisao": PROMOVIDO, "f1_necessario": necessario,
                "motivo": (f"F1_val do vencedor {f1_desafiante:.4f} >= F1_val do vigente "
                           f"{f1_vigente:.4f} + {margem:.2f} e >= mínimo {minimo:.2f}")}
    falhas = []
    if not supera_margem:
        falhas.append(f"< F1_val do vigente {f1_vigente:.4f} + margem {margem:.2f} "
                      f"(precisava de {f1_vigente + margem:.4f})")
    if not atinge_minimo:
        falhas.append(f"< mínimo {minimo:.2f}")
    return {"decisao": MANTIDO, "f1_necessario": necessario,
            "motivo": f"F1_val do vencedor {f1_desafiante:.4f} " + " e ".join(falhas)
                      + "; o vigente foi mantido"}


# --------------------------------------------------------------------------- #
# Manifesto
# --------------------------------------------------------------------------- #
def carregar_manifesto(cfg: dict | None = None, caminho: Path | None = None) -> dict | None:
    """``models/modelo_vigente.json`` ou ``None`` se ainda não há vigente."""
    p = Path(caminho) if caminho else caminho_manifesto(cfg)
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return m if m.get("run_id") else None


def avaliar_vigente(manifesto: dict, df: pd.DataFrame, cfg: dict | None = None) -> dict:
    """Carrega o vigente pelo ``run_id`` do manifesto e o avalia em ``df``."""
    modelo = carregar_modelo(manifesto["run_id"], cfg)
    try:
        return avaliar(modelo, df, cfg)
    except (KeyError, ValueError) as e:
        raise RuntimeError(
            f"o vigente (run {manifesto['run_id']}) não pôde ser avaliado nas partições atuais: {e}. "
            "Provavelmente o schema mudou (coluna removida). O pipeline para aqui, sem promover."
        ) from e


# --------------------------------------------------------------------------- #
# 3. Ciclo completo de promoção (sem efeitos colaterais em disco)
# --------------------------------------------------------------------------- #
def executar_promocao(competicao: dict, validacao: pd.DataFrame, teste: pd.DataFrame,
                      cfg: dict | None = None, manifesto: dict | None = ...) -> dict:
    """Escolhe o vencedor, compara com o vigente e calcula o novo manifesto.

    Não grava nada: devolve um dict com ``vencedor``, ``vigente_anterior``,
    ``decisao``, ``manifesto`` (novo conteúdo) e ``linha_historico``. Quem grava
    é ``gravar_decisao``, chamado no fim do notebook 04.

    ``manifesto=...`` (padrão) lê do disco; passe ``None`` para simular "sem vigente".
    """
    cfg = cfg or carregar_config()
    regra = carregar_regra_promocao(cfg)
    candidatos = competicao["candidatos"]
    nome_venc = escolher_vencedor(candidatos, regra)
    venc = candidatos[nome_venc]
    f1_venc = float(venc["metricas_validacao"]["f1"])

    anterior = carregar_manifesto(cfg) if manifesto is ... else manifesto
    met_val_anterior = avaliar_vigente(anterior, validacao, cfg) if anterior else None
    f1_anterior = met_val_anterior["f1"] if met_val_anterior else None

    dec = decidir(f1_venc, f1_anterior, regra)
    promovido = dec["decisao"] == PROMOVIDO
    agora = datetime.now(timezone.utc).isoformat(timespec="seconds")

    if promovido:
        novo = {
            "run_id": venc["run_id"], "model_uri": model_uri(venc["run_id"]),
            "modelo": nome_venc, "algoritmo": venc["algoritmo"],
            "features": list(competicao["features"]), "n_features": len(competicao["features"]),
            "treinado_no_ciclo": competicao["cycle_id"],
            "treinado_com_data_version": competicao["data_version"],
            "promovido_no_ciclo": competicao["cycle_id"], "promovido_em": agora,
            "metricas_validacao": venc["metricas_validacao"],
            "registry": None,
        }
    elif anterior:
        novo = {k: anterior.get(k) for k in (
            "run_id", "model_uri", "modelo", "algoritmo", "features", "n_features",
            "treinado_no_ciclo", "treinado_com_data_version", "promovido_no_ciclo",
            "promovido_em", "registry")}
        novo["metricas_validacao"] = met_val_anterior
    else:
        novo = None   # ciclo 1 abaixo do mínimo: continua sem vigente

    # teste: só agora, só do vigente resultante, só para o relatório
    if novo:
        novo["metricas_teste"] = avaliar(carregar_modelo(novo["run_id"], cfg), teste, cfg)

    manifesto_novo = {
        "cycle_id": competicao["cycle_id"],
        "data_version": competicao["data_version"],
        "decisao": dec["decisao"],
        "motivo": dec["motivo"],
        "decidido_em": agora,
        # --- modelo vigente após a decisão -------------------------------------
        "run_id": novo["run_id"] if novo else None,
        "model_uri": novo["model_uri"] if novo else None,
        "modelo": novo["modelo"] if novo else None,
        "algoritmo": novo["algoritmo"] if novo else None,
        "features": novo["features"] if novo else [],
        "n_features": novo["n_features"] if novo else 0,
        "f1_validacao": novo["metricas_validacao"]["f1"] if novo else None,
        "f1_teste": novo["metricas_teste"]["f1"] if novo else None,
        "metricas_validacao": novo["metricas_validacao"] if novo else None,
        "metricas_teste": novo["metricas_teste"] if novo else None,
        "treinado_no_ciclo": novo["treinado_no_ciclo"] if novo else None,
        "treinado_com_data_version": novo["treinado_com_data_version"] if novo else None,
        "promovido_no_ciclo": novo["promovido_no_ciclo"] if novo else None,
        "promovido_em": novo["promovido_em"] if novo else None,
        "registry": novo.get("registry") if novo else None,
        # --- contexto da decisão -----------------------------------------------
        "desafiante": {"modelo": nome_venc, "algoritmo": venc["algoritmo"], "run_id": venc["run_id"],
                       "f1_validacao": f1_venc},
        "candidatos": {n: {"algoritmo": c["algoritmo"], "run_id": c["run_id"],
                           "f1_validacao": float(c["metricas_validacao"]["f1"])}
                       for n, c in candidatos.items()},
        "vigente_anterior": ({"run_id": anterior["run_id"], "modelo": anterior.get("modelo"),
                              "algoritmo": anterior.get("algoritmo"),
                              "treinado_no_ciclo": anterior.get("treinado_no_ciclo"),
                              "f1_validacao": f1_anterior,
                              "metricas_validacao": met_val_anterior}
                             if anterior else None),
        "f1_necessario": dec["f1_necessario"],
        "regra": regra,
        "decisao_run_id": None,
    }
    return {"vencedor": nome_venc, "decisao": dec, "manifesto": manifesto_novo,
            "vigente_anterior": anterior, "linha_historico": linha_historico(manifesto_novo)}


def linha_historico(m: dict) -> dict:
    cand = m["candidatos"]
    ant = m["vigente_anterior"] or {}
    return {
        "cycle_id": m["cycle_id"], "data_version": m["data_version"], "decidido_em": m["decidido_em"],
        "f1_val_A": cand.get("A", {}).get("f1_validacao"),
        "f1_val_B": cand.get("B", {}).get("f1_validacao"),
        "vencedor_modelo": m["desafiante"]["modelo"],
        "vencedor_algoritmo": m["desafiante"]["algoritmo"],
        "vencedor_run_id": m["desafiante"]["run_id"],
        "f1_val_vencedor": m["desafiante"]["f1_validacao"],
        "vigente_anterior_run_id": ant.get("run_id"),
        "vigente_anterior_algoritmo": ant.get("algoritmo"),
        "f1_val_vigente_anterior": ant.get("f1_validacao"),
        "f1_necessario": m["f1_necessario"],
        "decisao": m["decisao"], "motivo": m["motivo"],
        "vigente_run_id": m["run_id"], "vigente_modelo": m["modelo"],
        "vigente_algoritmo": m["algoritmo"], "vigente_treinado_no_ciclo": m["treinado_no_ciclo"],
        "f1_val_vigente": m["f1_validacao"], "f1_teste_vigente": m["f1_teste"],
        "decisao_run_id": m["decisao_run_id"],
    }


# --------------------------------------------------------------------------- #
# 4. MLflow: run da decisão (+ Model Registry quando promove)
# --------------------------------------------------------------------------- #
def registrar_decisao_mlflow(resultado: dict, cfg: dict | None = None,
                             figuras: dict | None = None, usar_registry: bool = True) -> str:
    """Um run ``promocao`` por ciclo com a decisão; em caso de promoção, registra
    o modelo no Model Registry e move o alias ``vigente`` para ele.

    Atualiza ``resultado['manifesto']`` com ``decisao_run_id`` e ``registry``.
    """
    cfg = cfg or carregar_config()
    mlflow = configurar_mlflow(cfg)
    m = resultado["manifesto"]
    ant = m["vigente_anterior"] or {}
    with mlflow.start_run(run_name=f"{ETAPA}_{m['cycle_id']}") as run:
        mlflow.set_tags({"cycle_id": m["cycle_id"], "data_version": m["data_version"],
                         "etapa": ETAPA, "decisao": m["decisao"],
                         "vencedor_run_id": m["desafiante"]["run_id"],
                         "vigente_run_id": m["run_id"] or "",
                         "vigente_anterior_run_id": ant.get("run_id") or "",
                         "algoritmo": m["algoritmo"] or ""})
        mlflow.log_params({f"regra.{k}": v for k, v in m["regra"].items()}
                          | {"vencedor": m["desafiante"]["modelo"],
                             "vencedor_algoritmo": m["desafiante"]["algoritmo"]})
        metricas = {"promovido": int(m["decisao"] == PROMOVIDO),
                    "f1_val_vencedor": m["desafiante"]["f1_validacao"],
                    "f1_necessario": m["f1_necessario"]}
        for n, c in m["candidatos"].items():
            metricas[f"f1_val_{n}"] = c["f1_validacao"]
        if ant.get("f1_validacao") is not None:
            metricas["f1_val_vigente_anterior"] = ant["f1_validacao"]
        if m["f1_validacao"] is not None:
            metricas["f1_val_vigente"] = m["f1_validacao"]
            metricas["f1_teste_vigente"] = m["f1_teste"]
        mlflow.log_metrics(metricas)
        for caminho, fig in (figuras or {}).items():
            mlflow.log_figure(fig, caminho)

        if usar_registry and m["decisao"] == PROMOVIDO:
            m["registry"] = _registrar_no_registry(mlflow, m, cfg)
        m["decisao_run_id"] = run.info.run_id
        mlflow.log_dict(m, ARQ_DECISAO)
    resultado["linha_historico"] = linha_historico(m)
    return run.info.run_id


def _registrar_no_registry(mlflow, m: dict, cfg: dict) -> dict | None:
    """Best-effort: o manifesto é a fonte da verdade; o registry é um espelho na UI."""
    nome = cfg.get("promocao", {}).get("registry_nome", "fraude_ethereum")
    try:
        from mlflow import MlflowClient

        versao = mlflow.register_model(m["model_uri"], nome)
        cliente = MlflowClient()
        cliente.set_registered_model_alias(nome, ALIAS_VIGENTE, versao.version)
        cliente.set_model_version_tag(nome, versao.version, "cycle_id", m["cycle_id"])
        return {"nome": nome, "versao": str(versao.version), "alias": ALIAS_VIGENTE}
    except Exception as e:  # noqa: BLE001 — falha no registry não deve derrubar a decisão
        print(f"[aviso] não foi possível registrar no Model Registry: {e}")
        return None


# --------------------------------------------------------------------------- #
# 5. Gravação (último passo do notebook 04)
# --------------------------------------------------------------------------- #
def gravar_decisao(resultado: dict, cfg: dict | None = None, caminho_man: Path | None = None,
                   caminho_hist: Path | None = None, dir_ciclo: Path | None = None) -> dict:
    """Escreve ``modelo_vigente.json``, acrescenta ao ``historico_decisoes.csv``
    e salva ``artifacts/<cycle_id>/decisao.json``.

    O histórico tem uma linha por ciclo: rodar de novo o mesmo ``cycle_id``
    substitui a linha dele em vez de duplicar.
    """
    cfg = cfg or carregar_config()
    m = resultado["manifesto"]
    man = Path(caminho_man) if caminho_man else caminho_manifesto(cfg)
    hist = Path(caminho_hist) if caminho_hist else caminho_historico(cfg)
    pasta = Path(dir_ciclo) if dir_ciclo else dir_artifacts(cfg) / m["cycle_id"]
    for p in (man.parent, hist.parent, pasta):
        p.mkdir(parents=True, exist_ok=True)

    texto = json.dumps(m, indent=2, ensure_ascii=False, default=float) + "\n"
    (pasta / ARQ_DECISAO).write_text(texto, encoding="utf-8")

    linha = pd.DataFrame([resultado["linha_historico"]], columns=COLUNAS_HISTORICO)
    if hist.exists():
        antigo = pd.read_csv(hist)
        antigo = antigo.loc[antigo["cycle_id"].astype(str) != str(m["cycle_id"])]
        linha = pd.concat([antigo, linha], ignore_index=True)
    linha.to_csv(hist, index=False, lineterminator="\n")

    # o manifesto é o último arquivo escrito: é ele que o notebook 05 e o painel leem
    tmp = man.with_suffix(".json.tmp")
    tmp.write_text(texto, encoding="utf-8")
    tmp.replace(man)
    return {"manifesto": man, "historico": hist, "decisao": pasta / ARQ_DECISAO}


def carregar_historico(cfg: dict | None = None) -> pd.DataFrame:
    p = caminho_historico(cfg)
    return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=COLUNAS_HISTORICO)
