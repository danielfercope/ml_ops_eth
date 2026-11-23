"""Seleção do vencedor e decisão de promoção (Dev 3).

Fluxo do notebook 04, a cada ciclo:

    1. ``escolher_vencedor``  -- entre A e B, maior F1 de VALIDAÇÃO
       (empate -> ``promocao.desempate_vencedor`` do config, o modelo A).
    2. ``carregar_manifesto`` -- lê ``models/modelo_vigente.json``; o vigente é
       carregado do MLflow pelo ``run_id`` e reavaliado na MESMA validação do
       ciclo atual (os dados mudam entre ciclos, a régua tem de ser a mesma).
    3. ``verificar_qualidade`` -- verificações do vencedor além do limiar
       (``promocao.verificacoes_qualidade``): precision e recall mínimos, modelo
       não degenerado (taxa prevista de fraude dentro de uma faixa), métricas
       finitas e recarga do MLflow reproduzindo o F1 de validação.
    4. ``decidir``            -- aplica a regra fixada no config:
           promove  se  F1_val >= F1_vigente + margem_minima  e  F1_val >= f1_minimo
           sem vigente (ciclo 1): promove se F1_val >= f1_minimo
           e, em ambos os casos, se o vencedor passou nas verificações de qualidade
       senão o vigente é mantido.
    5. Só DEPOIS da decisão o vigente resultante é medido no teste (número do
       relatório); o teste nunca participa da escolha.
    6. ``gravar_decisao``     -- escreve o manifesto e o histórico. É o último
       passo do notebook: se algo falhar antes, nada muda no disco.

Modelos anteriores nunca são apagados: os runs ficam no MLflow e o manifesto
só troca o ponteiro (``run_id``).
"""
from __future__ import annotations

import json
import math
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
    "qualidade_aprovada", "qualidade_falhas",
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


def carregar_verificacoes_qualidade(cfg: dict | None = None) -> dict:
    """Limites das verificações de qualidade (``promocao.verificacoes_qualidade``)."""
    cfg = cfg or carregar_config()
    p = cfg.get("promocao", {}).get("verificacoes_qualidade")
    if not isinstance(p, dict):
        raise ValueError("config.yaml sem 'promocao.verificacoes_qualidade'")
    v = {
        "precision_minima": float(p.get("precision_minima", 0.0)),
        "recall_minimo": float(p.get("recall_minimo", 0.0)),
        "taxa_prevista_min": float(p.get("taxa_prevista_min", 0.0)),
        "taxa_prevista_max": float(p.get("taxa_prevista_max", 1.0)),
        "recarregar_do_mlflow": bool(p.get("recarregar_do_mlflow", True)),
        "tolerancia_recarga": float(p.get("tolerancia_recarga", 1e-6)),
    }
    for k in ("precision_minima", "recall_minimo", "taxa_prevista_min", "taxa_prevista_max"):
        if not 0 <= v[k] <= 1:
            raise ValueError(f"verificacoes_qualidade.{k} deve estar em [0, 1]")
    if v["taxa_prevista_min"] > v["taxa_prevista_max"]:
        raise ValueError("taxa_prevista_min > taxa_prevista_max")
    if v["tolerancia_recarga"] < 0:
        raise ValueError("tolerancia_recarga deve ser >= 0")
    return v


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
# 2. Verificações de qualidade do vencedor
# --------------------------------------------------------------------------- #
def _checagem(nome: str, ok: bool, valor, limite, descricao: str) -> dict:
    return {"nome": nome, "ok": bool(ok), "valor": valor, "limite": limite, "descricao": descricao}


def checar_metricas(met: dict, verif: dict) -> list[dict]:
    """Verificações sobre as métricas de validação do candidato (sem I/O)."""
    out = []
    nao_finitas = [k for k in ("f1", "precision", "recall")
                   if not isinstance(met.get(k), (int, float)) or not math.isfinite(met[k])]
    out.append(_checagem("metricas_finitas", not nao_finitas, nao_finitas or None, None,
                         f"métricas não finitas: {', '.join(nao_finitas)}"))
    if nao_finitas:
        return out
    p, r = float(met["precision"]), float(met["recall"])
    out.append(_checagem("precision_minima", p + EPS >= verif["precision_minima"], p, verif["precision_minima"],
                         f"precision {p:.4f} < mínima {verif['precision_minima']:.2f}"))
    out.append(_checagem("recall_minimo", r + EPS >= verif["recall_minimo"], r, verif["recall_minimo"],
                         f"recall {r:.4f} < mínimo {verif['recall_minimo']:.2f}"))
    n = int(met.get("n") or 0)
    if n > 0 and "tp" in met and "fp" in met:
        taxa = (int(met["tp"]) + int(met["fp"])) / n
        lo, hi = verif["taxa_prevista_min"], verif["taxa_prevista_max"]
        out.append(_checagem("taxa_prevista", lo - EPS <= taxa <= hi + EPS, taxa, [lo, hi],
                             f"taxa prevista de fraude {taxa:.4f} fora de [{lo:.2f}, {hi:.2f}] "
                             "(modelo degenerado)"))
    else:
        out.append(_checagem("taxa_prevista", False, None, None,
                             "matriz de confusão ausente: não dá para checar a taxa prevista"))
    return out


def checar_recarga(run_id: str, validacao: pd.DataFrame, f1_esperado: float, verif: dict,
                   cfg: dict | None = None) -> dict:
    """Recarrega o candidato do MLflow e confere se reproduz o F1 de validação."""
    try:
        f1 = float(avaliar(carregar_modelo(run_id, cfg), validacao, cfg)["f1"])
    except Exception as e:  # noqa: BLE001 — qualquer falha de carga reprova a promoção
        return _checagem("recarga_mlflow", False, None, verif["tolerancia_recarga"],
                         f"não foi possível recarregar o run {run_id} do MLflow: {e}")
    dif = abs(f1 - float(f1_esperado))
    return _checagem("recarga_mlflow", dif <= verif["tolerancia_recarga"], f1, f1_esperado,
                     f"modelo recarregado do MLflow tem F1_val {f1:.6f} != {float(f1_esperado):.6f}")


def verificar_qualidade(candidato: dict, validacao: pd.DataFrame | None = None, cfg: dict | None = None,
                        verif: dict | None = None) -> dict:
    """``{aprovada, checagens, falhas}`` do candidato vencedor.

    ``candidato`` no formato do ``competicao.json`` (``run_id`` + ``metricas_validacao``).
    A recarga do MLflow só roda se ``recarregar_do_mlflow`` e ``validacao`` for dada.
    """
    verif = verif or carregar_verificacoes_qualidade(cfg)
    met = candidato["metricas_validacao"]
    checagens = checar_metricas(met, verif)
    if verif["recarregar_do_mlflow"] and validacao is not None:
        checagens.append(checar_recarga(candidato["run_id"], validacao, met["f1"], verif, cfg))
    falhas = [c["descricao"] for c in checagens if not c["ok"]]
    return {"aprovada": not falhas, "checagens": checagens, "falhas": falhas, "limites": verif}


# --------------------------------------------------------------------------- #
# 3. Decisão
# --------------------------------------------------------------------------- #
def decidir(f1_desafiante: float, f1_vigente: float | None, regra: dict | None = None,
            qualidade: dict | None = None) -> dict:
    """Aplica a regra de promoção e devolve ``{decisao, motivo, f1_necessario}``.

    ``qualidade`` (de ``verificar_qualidade``): se reprovada, nada é promovido,
    mesmo que o F1 atinja o limiar.
    """
    d = _decidir_limiar(f1_desafiante, f1_vigente, regra or carregar_regra_promocao())
    if d["decisao"] == PROMOVIDO and qualidade is not None and not qualidade["aprovada"]:
        fim = "o vigente foi mantido" if f1_vigente is not None else "nada foi promovido"
        d = {**d, "decisao": MANTIDO,
             "motivo": (f"o vencedor atinge o limiar de F1 ({f1_desafiante:.4f} >= {d['f1_necessario']:.4f}), "
                        f"mas reprovou nas verificações de qualidade: {'; '.join(qualidade['falhas'])}; {fim}")}
    return d


def _decidir_limiar(f1_desafiante: float, f1_vigente: float | None, regra: dict) -> dict:
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
# 4. Ciclo completo de promoção (sem efeitos colaterais em disco)
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

    qualidade = verificar_qualidade(venc, validacao, cfg)
    dec = decidir(f1_venc, f1_anterior, regra, qualidade)
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
        "qualidade": qualidade,
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
        "qualidade_aprovada": (m.get("qualidade") or {}).get("aprovada"),
        "qualidade_falhas": "; ".join((m.get("qualidade") or {}).get("falhas") or []),
    }


# --------------------------------------------------------------------------- #
# 5. MLflow: run da decisão (+ Model Registry quando promove)
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
    q = m.get("qualidade")
    with mlflow.start_run(run_name=f"{ETAPA}_{m['cycle_id']}") as run:
        mlflow.set_tags({"cycle_id": m["cycle_id"], "data_version": m["data_version"],
                         "etapa": ETAPA, "decisao": m["decisao"],
                         "vencedor_run_id": m["desafiante"]["run_id"],
                         "vigente_run_id": m["run_id"] or "",
                         "vigente_anterior_run_id": ant.get("run_id") or "",
                         "algoritmo": m["algoritmo"] or ""}
                        | ({"qualidade_aprovada": str(q["aprovada"]).lower()} if q else {}))
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
        if q:
            mlflow.log_params({f"qualidade.{k}": v for k, v in q["limites"].items()})
            metricas["qualidade_aprovada"] = int(q["aprovada"])
            for c in q["checagens"]:
                metricas[f"qualidade.{c['nome']}"] = int(c["ok"])
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
# 6. Gravação (último passo do notebook 04)
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
