#!/usr/bin/env bash
# CheXpert-plus-RRG impression valid, N=202. This is the official transfer split.
#
#   sbatch --export=ALL,MODEL_PATH=...,OUTPUT_PATH=...,EVAL_TAG=... \
#     scripts/sbatch_eval_chexpert_rrg_impression_l40.sh

#SBATCH --job-name=cxp_rrg_imp202
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MVP_ROOT="${MVP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs}"
MEDEVALKIT_ROOT="${MEDEVALKIT_ROOT:-/path/to/MedEvalKit}"
CONDA_SH="${CONDA_SH:-/share/apps/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-${HOME}/.conda/envs}"
CONDA_NAME_EVAL="${CONDA_NAME_EVAL:-qwencomp}"
CONDA_NAME_METRICS="${CONDA_NAME_METRICS:-medevalkit}"
PROCESSOR_SRC="${PROCESSOR_SRC:-/path/to/Qwen3-VL-2B-Instruct}"

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
OUTPUT_PATH="${OUTPUT_PATH:?set OUTPUT_PATH}"
EVAL_TAG="${EVAL_TAG:-cxp_rrg_imp202}"

mkdir -p "${MVP_ROOT}/slurm_logs" "${OUTPUT_PATH}"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

for f in preprocessor_config.json video_preprocessor_config.json processor_config.json; do
  if [[ ! -f "${MODEL_PATH}/${f}" && -f "${PROCESSOR_SRC}/${f}" ]]; then
    cp -n "${PROCESSOR_SRC}/${f}" "${MODEL_PATH}/${f}" || true
  fi
done

source "${CONDA_SH}"
conda activate "${CONDA_NAME_EVAL}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CONDA_NAME="${CONDA_NAME_EVAL}"
export MODEL_NAME="Qwen3-VL"
export MODEL_PATH
export OUTPUT_PATH
export EVAL_DATASETS="CheXpert_Plus"
export DATASETS_PATH="${MEDEVALKIT_ROOT}/utils/CheXpert_Plus_RRG_impression"
export MAX_EVAL_SAMPLES=0
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MIMIC_EVAL_BATCH_SIZE="${MIMIC_EVAL_BATCH_SIZE:-64}"
export ENABLE_PCA_VIZ=False
export ENABLE_CKNNA_VIZ=False
export USE_VLLM=False
echo "[INFO] tag=${EVAL_TAG} n=202 RRG impression"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUTPUT_PATH=${OUTPUT_PATH}"
cd "${MEDEVALKIT_ROOT}"
bash "${MEDEVALKIT_ROOT}/sbatch_eval_qwen3vl_mimic_cxr_frontal_l40.sh"

RESULT_DIR="${OUTPUT_PATH}/Qwen3-VL/CheXpert_Plus"
conda activate "${CONDA_NAME_METRICS}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
if [[ -f "${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6.0.29" ]]; then
  export LD_PRELOAD="${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6.0.29"
elif [[ -f "${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6" ]]; then
  export LD_PRELOAD="${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib/libstdc++.so.6"
else
  unset LD_PRELOAD || true
fi
export LD_LIBRARY_PATH="${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib:/share/apps/cuda-11.8/lib64:/share/apps/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MEDEVALKIT_ROOT
export SKIP_BERT_SCORE="${SKIP_BERT_SCORE:-0}"
export REPORT_METRICS_MODEL_RESULTS_ROOT="${OUTPUT_PATH}/Qwen3-VL"
export REPORT_METRICS_DATASETS="CheXpert_Plus"
cd "${MEDEVALKIT_ROOT}/utils/Metrics_Compute"
python cal_report_metrics.py
METRICS_JSON="${RESULT_DIR}/metrics_2.json"
if [[ ! -f "${METRICS_JSON}" ]]; then
  METRICS_JSON="${RESULT_DIR}/metrics.json"
fi

REPORTS_JSON="${RESULT_DIR}/reports_for_chexbert_ce.json"
CE_JSON="${RESULT_DIR}/chexbert_ce.json"
python - "${RESULT_DIR}/results.json" "${REPORTS_JSON}" <<'PY'
import json, sys
rows = json.load(open(sys.argv[1], encoding="utf-8"))
out = []
for row in rows:
    pred = row.get("response", "")
    if isinstance(pred, dict):
        pred = pred.get("content", "")
    pred = str(pred or "").strip()
    findings = str(row.get("findings") or "").strip()
    impression = str(row.get("impression") or "").strip()
    gt = f"Findings: {findings} Impression: {impression}.".strip()
    if gt and pred:
        out.append({"gt": gt, "pred": pred, "dataset": "CheXpert_Plus", "id": row.get("id", "")})
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OK] wrote {len(out)} pairs")
PY

cd "${MEDEVALKIT_ROOT}/PromptMRG"
export PROMPTMRG_BERT_PATH="${PROMPTMRG_BERT_PATH:-checkpoints/bert-base-uncased}"
python scripts/compute_chexbert_ce.py --input "${REPORTS_JSON}" --device cuda --out "${CE_JSON}"

python - "${METRICS_JSON}" "${CE_JSON}" "${RESULT_DIR}/metrics_all.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1], encoding="utf-8"))
ce = json.load(open(sys.argv[2], encoding="utf-8"))
ce_all = ce.get("all", ce.get("overall", ce))
src = metrics.get("nlg_and_ce", metrics)
for key in ("ce_precision", "ce_recall", "ce_f1", "ce_num_examples"):
    if key in ce_all:
        src[key] = ce_all[key]
        metrics[key] = ce_all[key]
if "n" in ce_all:
    src["ce_num_examples"] = ce_all["n"]
    metrics["n_evaluated"] = ce_all["n"]
metrics["ce_source"] = "chexbert_ce.json"
if "nlg_and_ce" in metrics:
    metrics["nlg_and_ce"] = src
json.dump(metrics, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(metrics, open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
keep = ["n_evaluated","rougeL","rouge1","bleu1","bleu2","meteor","rate","ce_f1","ce_precision","ce_recall"]
show = metrics.get("nlg_and_ce", metrics)
print({k: (round(show[k],4) if isinstance(show.get(k), float) else show.get(k)) for k in keep})
PY

echo "[DONE] ${RESULT_DIR}/metrics_all.json"
