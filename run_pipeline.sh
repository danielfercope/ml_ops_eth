#!/usr/bin/env bash
# =============================================================================
# Pipeline MLOps — um ciclo completo: 01 -> 02 -> 03 -> 04 -> 05
#
# Usado pelo workflow agendado (.github/workflows/pipeline_mlops.yml) e por
# quem roda localmente, para que os dois executem exatamente a mesma coisa.
#
#   ./run_pipeline.sh                 # ciclo local (CYCLE_ID = AAAAMMDD-HHMM-local)
#   CYCLE_ID=20261110-0900-r12 ./run_pipeline.sh
#
# Para no primeiro notebook que falhar (set -e + nbconvert --execute). Os
# notebooks executados vão para evidencias/<cycle_id>/; o manifesto do modelo
# vigente só é escrito no fim do 04, então uma falha antes disso não promove nada.
# =============================================================================
set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RAIZ"

PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null 2>&1 || PYTHON=python3

NOTEBOOKS=(
  notebooks/01_etl.ipynb
  notebooks/02_votacao_features.ipynb
  notebooks/03_treino_competicao.ipynb
  notebooks/04_selecao_promocao.ipynb
  notebooks/05_previsoes.ipynb
)
TIMEOUT="${NB_TIMEOUT:-1800}"   # segundos por célula

# --- cycle_id único para todos os notebooks deste ciclo ----------------------
export CYCLE_ID="${CYCLE_ID:-$("$PYTHON" -m src.rastreio cycle-id)}"
export PYTHONPATH="$RAIZ${PYTHONPATH:+:$PYTHONPATH}"
export MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR=false
export PYTHONHASHSEED=0

EVID="evidencias/$CYCLE_ID"
mkdir -p "$EVID"
LOG="$EVID/pipeline.log"
: > "$LOG"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "ciclo $CYCLE_ID — raiz $RAIZ"

for nb in "${NOTEBOOKS[@]}"; do
  [[ -f "$nb" ]] || { log "ERRO: $nb não existe"; exit 1; }
done

# --- MLflow: experimento com artefatos em mlartifacts/ e caminhos do mlflow.db
log "preparando o MLflow"
"$PYTHON" -m src.rastreio preparar >> "$LOG" 2>&1

# --- notebooks em sequência --------------------------------------------------
for nb in "${NOTEBOOKS[@]}"; do
  nome="$(basename "$nb")"
  inicio=$(date +%s)
  log "executando $nome"
  if ! "$PYTHON" -m jupyter nbconvert --to notebook --execute "$nb" \
        --output-dir "$EVID" --output "$nome" \
        --ExecutePreprocessor.timeout="$TIMEOUT" \
        --ExecutePreprocessor.kernel_name=python3 >> "$LOG" 2>&1; then
    log "FALHOU: $nome — veja $EVID/$nome e $LOG"
    exit 1
  fi
  log "ok: $nome ($(( $(date +%s) - inicio ))s)"
done

# --- resumo do ciclo (evidência) ---------------------------------------------
"$PYTHON" - "$EVID" <<'PY' | tee -a "$LOG"
import json, os, sys
from pathlib import Path
from src.rastreio import runs_do_ciclo, obter_data_version

evid = Path(sys.argv[1])
cid = os.environ["CYCLE_ID"]
man_p = Path("models/modelo_vigente.json")
man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
runs = runs_do_ciclo(cid)
resumo = {
    "cycle_id": cid,
    "data_version": obter_data_version(),
    "gatilho": os.environ.get("GITHUB_EVENT_NAME", "local"),
    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
    "github_run_url": (f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
                       f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
                       if os.environ.get("GITHUB_RUN_ID") else None),
    "commit": os.environ.get("GITHUB_SHA"),
    "decisao": man.get("decisao") if man.get("cycle_id") == cid else None,
    "motivo": man.get("motivo") if man.get("cycle_id") == cid else None,
    "vigente": {k: man.get(k) for k in ("run_id", "modelo", "algoritmo", "treinado_no_ciclo",
                                         "f1_validacao", "f1_teste")},
    "runs_mlflow": [
        {k: (v if isinstance(v, str) else None) for k, v in {
            "etapa": r.get("tags.etapa"), "modelo": r.get("tags.modelo"),
            "run_id": r["run_id"], "status": r["status"]}.items()}
        for _, r in runs.iterrows()
    ],
}
(evid / "resumo_ciclo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False, default=str) + "\n",
                                        encoding="utf-8")
print(f"decisão: {resumo['decisao']} | vigente: {resumo['vigente']['algoritmo']} "
      f"({resumo['vigente']['run_id']}) | runs no MLflow: {len(resumo['runs_mlflow'])}")
PY

log "ciclo $CYCLE_ID concluído"
