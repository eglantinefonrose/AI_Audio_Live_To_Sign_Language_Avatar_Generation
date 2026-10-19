#!/usr/bin/env bash
# Wrapper d'entraînement LoRA : horodate le run (début / fin / durée),
# enregistre un log complet dans finetune/logs/ et écrit la provenance de
# l'entraînement dans <ADAPTERS>/training_meta.json (date, durée, hyperparamètres,
# dataset, commit git, dernière val loss). Les datasets sont rangés par hash dans
# finetune/data/<train_sha>/ (snapshot_data.py) : par défaut on prend le plus
# récent ; DATA=finetune/data/<hash> pour en choisir un. Un dossier hors de
# finetune/data/ est d'abord copié et rangé : l'entraînement lit toujours la
# version rangée, donc exactement les données archivées.
# Ce fichier de provenance suit ensuite le modèle :
# register_ollama.py -> finetune/models/<tag>.json -> eval/results -> fiche HF.
#
# Usage :
#   ./finetune/train.sh
#   ITERS=200 NUM_LAYERS=4 ./finetune/train.sh     # surcharge des hyperparamètres
#   MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit ADAPTERS=finetune/adapters-7b \
#     GRAD_CHECKPOINT=1 ./finetune/train.sh          # 7B sur 16 Go
#
# Hyperparamètres surchargeables par variable d'environnement :
#   MODEL, DATA (défaut : dataset le plus récent de finetune/data/), ADAPTERS,
#   ITERS, NUM_LAYERS, LR
#   GRAD_CHECKPOINT=1 : gradient checkpointing (beaucoup moins de mémoire, ~20-30 %
#   plus lent) — recommandé au-delà de 3B sur 16 Go.
set -euo pipefail
cd "$(dirname "$0")/.."   # racine du projet

MODEL="${MODEL:-mlx-community/Qwen2.5-3B-Instruct-4bit}"
DATA="${DATA:-}"   # vide = dataset le plus récent de finetune/data/
ADAPTERS="${ADAPTERS:-finetune/adapters}"
ITERS="${ITERS:-300}"
NUM_LAYERS="${NUM_LAYERS:-8}"
LR="${LR:-1e-4}"
GRAD_CHECKPOINT="${GRAD_CHECKPOINT:-0}"
EXTRA_ARGS=()
[[ "$GRAD_CHECKPOINT" == "1" ]] && EXTRA_ARGS+=(--grad-checkpoint)

mkdir -p finetune/logs

# Dataset rangé par hash (vérifié, ou copié s'il vient d'ailleurs) : "<train_sha> <dossier>".
SNAP="$(uv run python finetune/snapshot_data.py ${DATA:+"$DATA"})"
TRAIN_SHA="${SNAP%% *}"
TRAIN_DATA="${SNAP#* }"
TS="$(date +%Y%m%d-%H%M%S)"
LOG="finetune/logs/train_${TS}.log"

START=$(date +%s)
STARTED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
{
  echo "=== Entraînement LoRA ==="
  echo "Début   : $(date '+%Y-%m-%d %H:%M:%S')"
  echo "Modèle  : $MODEL"
  echo "Données : $TRAIN_DATA (train_sha $TRAIN_SHA)${DATA:+  ← $DATA}"
  echo "Adaptateur : $ADAPTERS"
  echo "Params  : iters=$ITERS  num_layers=$NUM_LAYERS  lr=$LR  batch=1  mask-prompt  grad_checkpoint=$GRAD_CHECKPOINT"
  echo "---"
} | tee "$LOG"

# L'entraînement proprement dit (sortie dupliquée vers le log).
uv run mlx_lm.lora \
  --model "$MODEL" \
  --train --data "$TRAIN_DATA" --adapter-path "$ADAPTERS" \
  --mask-prompt --batch-size 1 --num-layers "$NUM_LAYERS" --iters "$ITERS" \
  --learning-rate "$LR" --steps-per-eval 50 --steps-per-report 25 \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
  2>&1 | tee -a "$LOG"

END=$(date +%s)
ENDED_AT="$(date '+%Y-%m-%dT%H:%M:%S%z')"
DUR=$((END - START))
{
  echo "---"
  echo "Fin     : $(date '+%Y-%m-%d %H:%M:%S')"
  printf 'Durée   : %dh%02dm%02ds\n' $((DUR/3600)) $(((DUR%3600)/60)) $((DUR%60))
} | tee -a "$LOG"

# Provenance de l'entraînement, à côté de l'adaptateur (publiée avec lui sur HF).
VAL_LOSS="$(grep -oE 'Val loss [0-9.]+' "$LOG" | tail -1 | awk '{print $3}' || true)"
STARTED_AT="$STARTED_AT" ENDED_AT="$ENDED_AT" DURATION_S="$DUR" VAL_LOSS="$VAL_LOSS" \
MODEL="$MODEL" DATA="$DATA" TRAIN_DATA="$TRAIN_DATA" TRAIN_SHA="$TRAIN_SHA" \
ADAPTERS="$ADAPTERS" ITERS="$ITERS" \
NUM_LAYERS="$NUM_LAYERS" LR="$LR" GRAD_CHECKPOINT="$GRAD_CHECKPOINT" LOG="$LOG" \
uv run python - <<'PY'
import json, os, subprocess
from pathlib import Path

env = os.environ
def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()

snapshot = Path(env["TRAIN_DATA"])   # copie figée réellement lue par l'entraînement
manifest = snapshot / "manifest.json"
meta = {
    "started_at": env["STARTED_AT"],
    "ended_at": env["ENDED_AT"],
    "duration_s": int(env["DURATION_S"]),
    "approximate": False,
    "base_model": env["MODEL"],
    "hyperparams": {"iters": int(env["ITERS"]), "num_layers": int(env["NUM_LAYERS"]),
                    "learning_rate": float(env["LR"]), "batch_size": 1, "mask_prompt": True,
                    "grad_checkpoint": env["GRAD_CHECKPOINT"] == "1"},
    "final_val_loss": float(env["VAL_LOSS"]) if env.get("VAL_LOSS") else None,
    "data_dir": env["DATA"] or str(snapshot),
    "train_sha": env["TRAIN_SHA"],
    "train_snapshot": str(snapshot),
    "dataset_manifest": json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None,
    "git_commit": git("rev-parse", "--short", "HEAD") or None,
    "git_dirty": bool(git("status", "--porcelain")),
    "log": env["LOG"],
}
out = Path(env["ADAPTERS"]) / "training_meta.json"
out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Provenance  : {out}")
PY
echo "Log complet : $LOG"
