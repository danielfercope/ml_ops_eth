"""Contrato dos JSONs do painel (Dev 5, critério 10).

Os cinco arquivos lidos pelo painel (``web/``) são descritos por JSON Schemas em
``web/dados/schemas/``. Este módulo valida uma pasta com esses arquivos em duas
camadas:

1. **Formato** — cada arquivo contra o seu schema (campos, tipos, padrões).
2. **Coerência** — regras que um schema não expressa: os mesmos ``cycle_id`` em
   todos os arquivos, eleitas da votação = features de A e B, vencedor = maior F1,
   decisão compatível com a regra de promoção, vigente.json = último ciclo, etc.

Uso:
    python -m src.contrato_painel web/dados/mocks/padrao   # valida uma pasta
    python -m src.contrato_painel public/data              # saída do exportador

``src/exportar_painel.py`` deve chamar ``validar_pasta`` antes de publicar.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from referencing import Registry, Resource

RAIZ = Path(__file__).resolve().parents[1]
DIR_SCHEMAS = RAIZ / "web" / "dados" / "schemas"
ARQUIVOS = ("ciclos", "votacao", "competicao", "vigente", "previsoes")
TOL = 1e-4  # métricas são arredondadas a 4 casas


def _registro() -> Registry:
    recursos = []
    for arq in DIR_SCHEMAS.glob("*.schema.json"):
        schema = json.loads(arq.read_text(encoding="utf-8"))
        recursos.append((schema["$id"], Resource.from_contents(schema)))
    return Registry().with_resources(recursos)


def validadores() -> dict[str, Draft202012Validator]:
    reg = _registro()
    out = {}
    for nome in ARQUIVOS:
        schema = json.loads((DIR_SCHEMAS / f"{nome}.schema.json").read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        out[nome] = Draft202012Validator(schema, registry=reg)
    return out


def validar_formato(dados: dict[str, dict]) -> list[str]:
    erros = []
    for nome, v in validadores().items():
        for e in sorted(v.iter_errors(dados[nome]), key=lambda e: list(e.absolute_path)):
            caminho = "/".join(map(str, e.absolute_path)) or "(raiz)"
            erros.append(f"{nome}.json: {caminho}: {e.message}")
    return erros


def _ids(lista: list[dict]) -> list[str]:
    return [c["cycle_id"] for c in lista]


def validar_coerencia(d: dict[str, dict]) -> list[str]:
    """Regras entre arquivos. Supõe que o formato já está válido."""
    erros: list[str] = []
    err = erros.append

    ciclos = d["ciclos"]["ciclos"]
    ids = _ids(ciclos)
    vot = {c["cycle_id"]: c for c in d["votacao"]["ciclos"]}
    comp = {c["cycle_id"]: c for c in d["competicao"]["ciclos"]}
    prev = {c["cycle_id"]: c for c in d["previsoes"]["ciclos"]}
    regra = d["vigente"]["regra_promocao"]
    metodos = [m["id"] for m in d["votacao"]["metodos"]]

    origens = {d[n]["origem"] for n in ARQUIVOS}
    if len(origens) > 1:
        err(f"arquivos misturam origens {sorted(origens)}")

    # ordem e unicidade
    if len(set(ids)) != len(ids):
        err("ciclos.json: cycle_id repetido")
    datas = [c["executado_em"] for c in ciclos]
    if datas != sorted(datas):
        err("ciclos.json: ciclos fora de ordem cronológica")
    for nome, mapa in (("votacao", vot), ("competicao", comp)):
        if _ids(d[nome]["ciclos"]) != ids:
            err(f"{nome}.json: cycle_ids diferentes de ciclos.json ({list(mapa)} != {ids})")
    if [i for i in _ids(d["previsoes"]["ciclos"]) if i not in ids]:
        err("previsoes.json: cycle_id que não existe em ciclos.json")

    vigente_corrente = None
    for c in ciclos:
        cid = c["cycle_id"]
        v, k = vot.get(cid), comp.get(cid)
        if v is None or k is None:
            continue

        # --- votação
        if v["data_version"] != c["data_version"] or k["data_version"] != c["data_version"]:
            err(f"{cid}: data_version diverge entre arquivos")
        tabela = v["tabela"]
        eleitas_tab = [l["feature"] for l in tabela if l["eleita"]]
        if eleitas_tab != v["eleitas"]:
            err(f"{cid}: votacao.eleitas difere das linhas com eleita=true")
        if len(v["eleitas"]) != c["n_features_eleitas"]:
            err(f"{cid}: n_features_eleitas ({c['n_features_eleitas']}) != {len(v['eleitas'])}")
        if len({l["feature"] for l in tabela}) != len(tabela):
            err(f"{cid}: feature repetida na tabela de votos")
        r = v["resumo"]
        if r["n_features_avaliadas"] != len(tabela) or r["n_eleitas"] != len(v["eleitas"]):
            err(f"{cid}: resumo da votação não bate com a tabela")
        n_feat = len(tabela)
        for l in tabela:
            f = l["feature"]
            if sorted(l["votos"]) != sorted(metodos) or sorted(l["ranks"]) != sorted(metodos):
                err(f"{cid}/{f}: votos/ranks não cobrem exatamente os métodos {metodos}")
                continue
            if sum(l["votos"].values()) != l["total_votos"]:
                err(f"{cid}/{f}: total_votos != soma dos votos")
            media = sum(l["ranks"].values()) / len(metodos)
            if abs(media - l["rank_medio"]) > 1e-3:
                err(f"{cid}/{f}: rank_medio {l['rank_medio']} != {media:.3f}")
            if any(rk > n_feat for rk in l["ranks"].values()):
                err(f"{cid}/{f}: rank maior que o nº de features")
            for m in metodos:
                if l["votos"][m] != int(l["ranks"][m] <= d["votacao"]["regra"]["top_k"]):
                    err(f"{cid}/{f}: voto de {m} incoerente com rank e top_k")
            if l["eleita"] != (l["motivo"] is not None):
                err(f"{cid}/{f}: motivo deve ser null se e somente se não eleita")
            if l["motivo"] == "votos_suficientes" and l["total_votos"] < d["votacao"]["regra"]["votos_minimos"]:
                err(f"{cid}/{f}: eleita por votos com menos que votos_minimos")
        for m in metodos:
            if sorted(l["ranks"][m] for l in tabela) != list(range(1, n_feat + 1)):
                err(f"{cid}: ranks de {m} não formam 1..{n_feat}")

        # --- competição
        papeis = [m["papel"] for m in k["modelos"]]
        esperado = ["A", "B"] + (["vigente"] if c["vigente_anterior"] else [])
        if papeis != esperado:
            err(f"{cid}: competicao.modelos deveria ter papéis {esperado}, tem {papeis}")
        mod = {m["papel"]: m for m in k["modelos"]}
        for p in ("A", "B"):
            if p in mod:
                if mod[p]["features"] != v["eleitas"]:
                    err(f"{cid}: modelo {p} não usa as features eleitas")
                if mod[p]["cycle_id_treino"] != cid:
                    err(f"{cid}: modelo {p} com cycle_id_treino diferente do ciclo")
                mc = mod[p]["matriz_confusao"]
                if sum(mc.values()) != k["n_validacao"] or k["n_validacao"] != c["dados"]["n_validacao"]:
                    err(f"{cid}: matriz de {p} não soma n_validacao")
        if "A" in mod and "B" in mod:
            melhor = "A" if mod["A"]["metricas"]["f1"] > mod["B"]["metricas"]["f1"] else "B"
            if abs(mod["A"]["metricas"]["f1"] - mod["B"]["metricas"]["f1"]) > TOL and melhor != k["vencedor"]:
                err(f"{cid}: vencedor {k['vencedor']} não tem o maior F1")
        if k["vencedor"] != c["vencedor"]["modelo"] or k["decisao"] != c["decisao"]:
            err(f"{cid}: vencedor/decisão divergem entre competicao.json e ciclos.json")
        venc = mod.get(c["vencedor"]["modelo"])
        if venc and (venc["run_id"] != c["vencedor"]["run_id"]
                     or abs(venc["metricas"]["f1"] - c["vencedor"]["f1_val"]) > TOL):
            err(f"{cid}: run_id/F1 do vencedor divergem entre arquivos")

        # --- promoção
        ant = c["vigente_anterior"]
        if vigente_corrente is None and ant is not None:
            err(f"{cid}: vigente_anterior preenchido sem vigente no ciclo anterior")
        if vigente_corrente is not None:
            if ant is None or ant["run_id"] != vigente_corrente["run_id"]:
                err(f"{cid}: vigente_anterior != vigente_apos do ciclo anterior")
            elif "vigente" in mod and (mod["vigente"]["run_id"] != ant["run_id"]
                                       or abs(mod["vigente"]["metricas"]["f1"] - ant["f1_val"]) > TOL):
                err(f"{cid}: modelo 'vigente' da competição diverge de vigente_anterior")
        f1v = c["vencedor"]["f1_val"]
        q = c["qualidade"]
        deveria = f1v >= regra["f1_minimo"] - TOL and (
            ant is None or f1v >= ant["f1_val"] + regra["margem_minima"] - TOL) and (
            q is None or q["aprovada"])
        if (c["decisao"] == "promovido") != deveria:
            err(f"{cid}: decisão '{c['decisao']}' contradiz a regra de promoção")
        if q is not None and q["aprovada"] == bool(q["falhas"]):
            err(f"{cid}: qualidade.aprovada deve ser true sse não há falhas")
        if c["decisao"] == "promovido":
            if c["vigente_apos"] is None or c["vigente_apos"]["run_id"] != c["vencedor"]["run_id"]:
                err(f"{cid}: promovido, mas vigente_apos não é o vencedor")
        elif (c["vigente_apos"] or {}).get("run_id") != (ant or {}).get("run_id"):
            err(f"{cid}: mantido, mas vigente_apos mudou")
        vigente_corrente = c["vigente_apos"]

        # --- previsões
        p = prev.get(cid)
        if vigente_corrente is not None and p is None:
            err(f"{cid}: havia vigente, mas não há previsões")
        if p is not None:
            if vigente_corrente is None or p["modelo"]["run_id"] != vigente_corrente["run_id"]:
                err(f"{cid}: previsões não usam o vigente do ciclo")
            if sum(p["matriz_confusao"].values()) != p["n_total"] or p["n_total"] != c["dados"]["n_teste"]:
                err(f"{cid}: matriz de previsões não soma n_total/n_teste")
            for l in p["amostra"]:
                if l["previsao"] != int(l["probabilidade"] >= p["limiar"]):
                    err(f"{cid}: previsão incoerente com probabilidade e limiar ({l['address']})")
                    break

    # --- vigente.json
    vg = d["vigente"]
    if not ciclos:
        if vg["vigente"] is not None or vg["ultimo_ciclo"] is not None or vg["modelo_mudou"]:
            err("vigente.json: sem ciclos, vigente/ultimo_ciclo devem ser null e modelo_mudou false")
        return erros
    ult = ciclos[-1]
    u = vg["ultimo_ciclo"]
    if u is None or u["cycle_id"] != ult["cycle_id"]:
        err("vigente.json: ultimo_ciclo não é o último de ciclos.json")
    else:
        if (u["decisao"], u["motivo"], u["data_version"]) != (ult["decisao"], ult["motivo"], ult["data_version"]):
            err("vigente.json: ultimo_ciclo diverge de ciclos.json")
        if u["vencedor"] != ult["vencedor"]:
            err("vigente.json: vencedor do último ciclo diverge de ciclos.json")
        f1_ant = (ult["vigente_anterior"] or {}).get("f1_val")
        if u["f1_vigente_anterior"] != f1_ant:
            err("vigente.json: f1_vigente_anterior diverge de ciclos.json")
    if (vg["vigente"] or {}).get("run_id") != (ult["vigente_apos"] or {}).get("run_id"):
        err("vigente.json: vigente != vigente_apos do último ciclo")
    if vg["modelo_mudou"] != (ult["decisao"] == "promovido"):
        err("vigente.json: modelo_mudou deve ser true sse o último ciclo promoveu")
    esperado_ant = ult["vigente_anterior"] if ult["decisao"] == "promovido" else None
    if (vg["anterior"] or {}).get("run_id") != (esperado_ant or {}).get("run_id"):
        err("vigente.json: 'anterior' deve ser o vigente substituído no último ciclo")
    if vg["vigente"] is not None:
        origem = next((c for c in ciclos if c["cycle_id"] == vg["vigente"]["cycle_id"]), None)
        if origem is None or origem["decisao"] != "promovido":
            err("vigente.json: vigente.cycle_id não é um ciclo que promoveu")
        elif vg["vigente"]["promovido_em"] != origem["executado_em"]:
            err("vigente.json: promovido_em != executado_em do ciclo de origem")
    return erros


def carregar_pasta(pasta: str | Path) -> dict[str, dict]:
    pasta = Path(pasta)
    faltando = [n for n in ARQUIVOS if not (pasta / f"{n}.json").exists()]
    if faltando:
        raise FileNotFoundError(f"{pasta}: faltam {', '.join(f + '.json' for f in faltando)}")
    return {n: json.loads((pasta / f"{n}.json").read_text(encoding="utf-8")) for n in ARQUIVOS}


def validar_pasta(pasta: str | Path) -> list[str]:
    """Lista de erros (vazia = válido). Coerência só roda se o formato passar."""
    dados = carregar_pasta(pasta)
    erros = validar_formato(dados)
    return erros or validar_coerencia(dados)


if __name__ == "__main__":
    alvos = sys.argv[1:] or [str(p) for p in sorted((RAIZ / "web/dados/mocks").iterdir())]
    falhou = False
    for alvo in alvos:
        erros = validar_pasta(alvo)
        print(f"{'OK ' if not erros else 'ERRO'} {alvo}")
        for e in erros:
            print(f"   - {e}")
        falhou |= bool(erros)
    sys.exit(1 if falhou else 0)
