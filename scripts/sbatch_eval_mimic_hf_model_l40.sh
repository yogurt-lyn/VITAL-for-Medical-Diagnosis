#!/usr/bin/env bash
# Eval already-merged HF model on MIMIC_CXRfrontal (NLG + CheXbert CE).
#
#   sbatch --export=ALL,MODEL_PATH=...,OUTPUT_PATH=... \
#     scripts/sbatch_eval_mimic_hf_model_l40.sh

#SBATCH --job-name=mimic_hf_eval
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
unset ROCR_VISIBLE_DEVICES

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDEVALKIT_ROOT="${MEDEVALKIT_ROOT:-/path/to/MedEvalKit}"
CONDA_SH="${CONDA_SH:?set CONDA_SH to your conda.sh path}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-${HOME}/.conda/envs}"
CONDA_NAME_EVAL="${CONDA_NAME_EVAL:-qwencomp}"
CONDA_NAME_METRICS="${CONDA_NAME_METRICS:-medevalkit}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
OUTPUT_PATH="${OUTPUT_PATH:?set OUTPUT_PATH}"
RESULT_DIR="${OUTPUT_PATH}/Qwen3-VL/MIMIC_CXRfrontal"

mkdir -p "${PROJECT_ROOT}/slurm_logs" "${OUTPUT_PATH}"
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[FATAL] missing MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
fi

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUTPUT_PATH=${OUTPUT_PATH}"

source "${CONDA_SH}"
conda activate "${CONDA_NAME_EVAL}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
unset PYTORCH_CUDA_ALLOC_CONF

cd "${MEDEVALKIT_ROOT}"
export CONDA_NAME="${CONDA_NAME_EVAL}"
export MODEL_NAME="Qwen3-VL"
export MODEL_PATH
export OUTPUT_PATH
export EVAL_DATASETS="MIMIC_CXRfrontal"
export DATASETS_PATH="${MEDEVALKIT_ROOT}/utils/MIMIC_CXRfrontal"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
export ENABLE_PCA_VIZ=False
export ENABLE_CKNNA_VIZ=False
export USE_VLLM=False
bash "${MEDEVALKIT_ROOT}/sbatch_eval_qwen3vl_mimic_cxr_frontal_l40.sh"

RESULTS_JSON="${RESULT_DIR}/results.json"
[[ -f "${RESULTS_JSON}" ]] || { echo "[FATAL] missing ${RESULTS_JSON}" >&2; exit 1; }

conda activate "${CONDA_NAME_METRICS}"
export PYTHONNOUSERSITE=1
if [[ -f "${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6.0.29" ]]; then
  export LD_PRELOAD="${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6.0.29"
elif [[ -f "${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6"
else
  unset LD_PRELOAD || true
fi
export LD_LIBRARY_PATH="${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export REPORT_METRICS_MODEL_RESULTS_ROOT="${OUTPUT_PATH}/Qwen3-VL"
export REPORT_METRICS_DATASETS="MIMIC_CXRfrontal"
export REPORT_METRICS_OUTPUT_NAME="metrics_conversations_gt.json"
python "${MEDEVALKIT_ROOT}/utils/Metrics_Compute/cal_report_metrics_conversations_gt.py"

REPORTS_JSON="${RESULT_DIR}/reports_for_chexbert_ce.json"
CE_JSON="${RESULT_DIR}/chexbert_ce.json"
python - "${RESULTS_JSON}" "${REPORTS_JSON}" <<'PY'
import json, sys
results_path, out_path = sys.argv[1], sys.argv[2]
rows = json.load(open(results_path, encoding="utf-8"))
records = []
for row in rows:
    gt = ""
    for msg in row.get("conversations", []) or []:
        if msg.get("from") == "gpt":
            gt = msg.get("value", "")
            break
    pred = row.get("response", "")
    if isinstance(pred, dict):
        pred = pred.get("content", "")
    if gt and pred:
        records.append({"gt": gt, "pred": pred, "dataset": "MIMIC_CXRfrontal", "id": row.get("id", "")})
json.dump(records, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OK] wrote {len(records)} pairs")
PY

cd "${MEDEVALKIT_ROOT}/PromptMRG"
export PROMPTMRG_BERT_PATH="${PROMPTMRG_BERT_PATH:-checkpoints/bert-base-uncased}"
python scripts/compute_chexbert_ce.py --input "${REPORTS_JSON}" --device cuda --out "${CE_JSON}"

python - "${RESULT_DIR}/metrics_conversations_gt.json" "${CE_JSON}" "${RESULT_DIR}" <<'PY'
import json, sys
from pathlib import Path
metrics_path, ce_path, result_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8")) if Path(metrics_path).is_file() else {}
ce = json.loads(Path(ce_path).read_text(encoding="utf-8"))
ce_all = ce.get("all", {})
for key in ("ce_precision", "ce_recall", "ce_f1", "ce_num_examples"):
    if key in ce_all:
        metrics[key] = ce_all[key]
if "n" in ce_all:
    metrics["ce_num_examples"] = ce_all["n"]
metrics["ce_source"] = "chexbert_ce.json"
Path(metrics_path).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
Path(result_dir / "metrics_all.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print("[OK] CE:", {k: metrics.get(k) for k in ("ce_precision", "ce_recall", "ce_f1", "ce_num_examples")})
PY

echo "[DONE] ${RESULT_DIR}/metrics_all.json"
