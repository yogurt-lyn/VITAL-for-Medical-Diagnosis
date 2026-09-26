#!/usr/bin/env bash
# One-dataset gen + NLG + CheXbert CE for a merged 2B HF checkpoint.
# Intended to be submitted 3x in parallel (MIMIC / IU / CheXpert) on different GPUs.
#
#   sbatch --export=ALL,CKPT_ROOT=...,EVAL_DATASET=...,DATA_SUBDIR=...,GT_MODE=...,OUTPUT_PATH=...,MAX_NEW_TOKENS=...,MIMIC_EVAL_BATCH_SIZE=... \
#     scripts/sbatch_eval_one_ds_qwen3vl_l40.sh

#SBATCH --job-name=eval_one_ds
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=128G
#SBATCH --time=24:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MVP_ROOT="${MVP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs}"
MEDEVALKIT_ROOT="${MEDEVALKIT_ROOT:-/path/to/MedEvalKit}"
CONDA_SH="${CONDA_SH:?set CONDA_SH to your conda.sh path}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-${HOME}/.conda/envs}"
CONDA_NAME_EVAL="${CONDA_NAME_EVAL:-qwencomp}"
CONDA_NAME_METRICS="${CONDA_NAME_METRICS:-medevalkit}"
PROCESSOR_SRC="${PROCESSOR_SRC:-/path/to/Qwen3-VL-2B-Instruct}"

EVAL_DATASET="${EVAL_DATASET:?set EVAL_DATASET}"
DATA_SUBDIR="${DATA_SUBDIR:?set DATA_SUBDIR}"
GT_MODE="${GT_MODE:?set GT_MODE}"  # conversations | findings_impression
OUTPUT_PATH="${OUTPUT_PATH:?set OUTPUT_PATH}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MIMIC_EVAL_BATCH_SIZE="${MIMIC_EVAL_BATCH_SIZE:-8}"

MODEL_PATH="${MODEL_PATH:-}"
if [[ -z "${MODEL_PATH}" ]]; then
  CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT or MODEL_PATH}"
  if [[ -f "${CKPT_ROOT}/.last_merged_hf" ]]; then
    MODEL_PATH="$(tr -d '\n' < "${CKPT_ROOT}/.last_merged_hf")"
  else
    MODEL_PATH=""
    for d in $(ls -d "${CKPT_ROOT}"/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
      if [[ -f "${d}/actor_merged_hf/config.json" ]]; then
        MODEL_PATH="${d}/actor_merged_hf"
      fi
    done
  fi
fi
if [[ -z "${MODEL_PATH}" || ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[FATAL] missing merged HF. MODEL_PATH='${MODEL_PATH:-}' CKPT_ROOT='${CKPT_ROOT:-}'" >&2
  exit 1
fi

mkdir -p "${MVP_ROOT}/slurm_logs" "${OUTPUT_PATH}"
unset ROCR_VISIBLE_DEVICES
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

for f in preprocessor_config.json video_preprocessor_config.json processor_config.json; do
  if [[ ! -f "${MODEL_PATH}/${f}" && -f "${PROCESSOR_SRC}/${f}" ]]; then
    cp -n "${PROCESSOR_SRC}/${f}" "${MODEL_PATH}/${f}" || true
  fi
done

echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-manual}"
echo "[INFO] EVAL_DATASET=${EVAL_DATASET} GT_MODE=${GT_MODE}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUTPUT_PATH=${OUTPUT_PATH}"
echo "[INFO] DATASETS_PATH=${MEDEVALKIT_ROOT}/utils/${DATA_SUBDIR}"
echo "[INFO] MAX_NEW_TOKENS=${MAX_NEW_TOKENS} BATCH=${MIMIC_EVAL_BATCH_SIZE}"

source "${CONDA_SH}"
conda activate "${CONDA_NAME_EVAL}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
unset LD_PRELOAD || true
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CONDA_NAME="${CONDA_NAME_EVAL}"
export MODEL_NAME="Qwen3-VL"
export MODEL_PATH
export OUTPUT_PATH
export EVAL_DATASETS="${EVAL_DATASET}"
export DATASETS_PATH="${MEDEVALKIT_ROOT}/utils/${DATA_SUBDIR}"
export MAX_EVAL_SAMPLES=0
export MAX_NEW_TOKENS
export MIMIC_EVAL_BATCH_SIZE
export ENABLE_PCA_VIZ=False
export ENABLE_CKNNA_VIZ=False
export USE_VLLM=False
cd "${MEDEVALKIT_ROOT}"
RESULT_DIR="${OUTPUT_PATH}/Qwen3-VL/${EVAL_DATASET}"
if [[ "${SKIP_GEN:-0}" == "1" && -f "${RESULT_DIR}/results.json" ]]; then
  echo "[STAGE gen] skip existing ${RESULT_DIR}/results.json"
else
  bash "${MEDEVALKIT_ROOT}/sbatch_eval_qwen3vl_mimic_cxr_frontal_l40.sh"
fi

if [[ ! -f "${RESULT_DIR}/results.json" ]]; then
  echo "[FATAL] missing ${RESULT_DIR}/results.json" >&2
  exit 1
fi

echo "[STAGE nlg+ce] ${EVAL_DATASET}"
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
export LD_LIBRARY_PATH="${CONDA_ENVS_DIR}/${CONDA_NAME_METRICS}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MEDEVALKIT_ROOT
export REPORT_METRICS_MODEL_RESULTS_ROOT="${OUTPUT_PATH}/Qwen3-VL"
export REPORT_METRICS_DATASETS="${EVAL_DATASET}"
cd "${MEDEVALKIT_ROOT}"
if [[ "${GT_MODE}" == "conversations" ]]; then
  export REPORT_METRICS_OUTPUT_NAME="metrics_conversations_gt.json"
  python "${MEDEVALKIT_ROOT}/utils/Metrics_Compute/cal_report_metrics_conversations_gt.py"
  METRICS_JSON="${RESULT_DIR}/metrics_conversations_gt.json"
else
  export SKIP_BERT_SCORE="${SKIP_BERT_SCORE:-0}"
  cd "${MEDEVALKIT_ROOT}/utils/Metrics_Compute"
  python cal_report_metrics.py
  if [[ -f "${RESULT_DIR}/metrics_2.json" ]]; then
    METRICS_JSON="${RESULT_DIR}/metrics_2.json"
  else
    METRICS_JSON="${RESULT_DIR}/metrics.json"
  fi
fi

REPORTS_JSON="${RESULT_DIR}/reports_for_chexbert_ce.json"
CE_JSON="${RESULT_DIR}/chexbert_ce.json"
python - "${RESULT_DIR}/results.json" "${REPORTS_JSON}" "${EVAL_DATASET}" "${GT_MODE}" <<'PY'
import json, sys
results_path, out_path, dataset, gt_mode = sys.argv[1:5]
rows = json.load(open(results_path, encoding="utf-8"))
records = []
for row in rows:
    pred = row.get("response", "")
    if isinstance(pred, dict):
        pred = pred.get("content", "")
    pred = str(pred or "").strip()
    gt = ""
    if gt_mode == "conversations":
        for msg in row.get("conversations") or []:
            if msg.get("from") in {"gpt", "assistant"}:
                gt = str(msg.get("value") or "").strip()
                if gt:
                    break
    else:
        findings = str(row.get("findings") or "").strip()
        impression = str(row.get("impression") or "").strip()
        if findings or impression:
            gt = f"Findings: {findings} Impression: {impression}.".strip()
        else:
            for msg in row.get("conversations") or []:
                if msg.get("from") in {"gpt", "assistant"}:
                    gt = str(msg.get("value") or "").strip()
                    if gt:
                        break
    if gt and pred:
        records.append({"gt": gt, "pred": pred, "dataset": dataset, "id": row.get("id", "")})
json.dump(records, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"[OK] wrote {len(records)} {dataset} CE pairs")
PY

cd "${MEDEVALKIT_ROOT}/PromptMRG"
export PROMPTMRG_BERT_PATH="${PROMPTMRG_BERT_PATH:-checkpoints/bert-base-uncased}"
python scripts/compute_chexbert_ce.py --input "${REPORTS_JSON}" --device cuda --out "${CE_JSON}"

python - "${METRICS_JSON}" "${CE_JSON}" "${RESULT_DIR}/metrics_all.json" <<'PY'
import json, sys
metrics_path, ce_path, all_path = sys.argv[1:4]
metrics = json.load(open(metrics_path, encoding="utf-8"))
ce = json.load(open(ce_path, encoding="utf-8"))
ce_all = ce.get("all", ce.get("overall", ce))
src = metrics.get("nlg_and_ce", metrics)
for key in ("ce_precision", "ce_recall", "ce_f1", "ce_num_examples"):
    if key in ce_all:
        src[key] = ce_all[key]
        metrics[key] = ce_all[key]
if "n" in ce_all:
    src["ce_num_examples"] = ce_all["n"]
    metrics["n_evaluated"] = ce_all["n"]
    metrics.setdefault("ce_num_examples", ce_all["n"])
metrics["ce_source"] = "chexbert_ce.json"
if "nlg_and_ce" in metrics:
    metrics["nlg_and_ce"] = src
json.dump(metrics, open(metrics_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
json.dump(metrics, open(all_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
keep = ["n_evaluated","rougeL","rouge1","bleu1","bleu2","meteor","rate","ce_f1","ce_precision","ce_recall"]
show = metrics.get("nlg_and_ce", metrics)
print({k: (round(show[k],4) if isinstance(show.get(k), float) else show.get(k)) for k in keep})
PY

echo "[DONE] ${RESULT_DIR}/metrics_all.json"
