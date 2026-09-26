#!/usr/bin/env bash
# Eval an already-merged HF model on NeoCXR + NeoCXR_EV (NLG + clinic metrics).
#
#   sbatch --export=ALL,MODEL_PATH=...,OUTPUT_TAG=mimic_headfp_w005 \
#     scripts/sbatch_eval_mimic_model_on_neocxr_ev_l40.sh

#SBATCH --job-name=mimic2neo_eval
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err
#SBATCH --partition=L40
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
unset ROCR_VISIBLE_DEVICES

MEDEVALKIT_ROOT="${MEDEVALKIT_ROOT:-/path/to/MedEvalKit}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
OUTPUT_TAG="${OUTPUT_TAG:?set OUTPUT_TAG}"
CONDA_SH="${CONDA_SH:-/share/apps/miniconda3/etc/profile.d/conda.sh}"
CONDA_NAME_EVAL="${CONDA_NAME_EVAL:-qwencomp}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "[FATAL] missing MODEL_PATH=${MODEL_PATH}" >&2
  exit 1
fi

OUT_NEO="${MEDEVALKIT_ROOT}/eval_results/mimic_models_on_neocxr/${OUTPUT_TAG}"
OUT_EV="${MEDEVALKIT_ROOT}/eval_results/mimic_models_on_neocxr_ev/${OUTPUT_TAG}"

echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] OUT_NEO=${OUT_NEO}"
echo "[INFO] OUT_EV=${OUT_EV}"

source "${CONDA_SH}"
conda activate "${CONDA_NAME_EVAL}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
unset PYTORCH_CUDA_ALLOC_CONF

export MODEL_NAME="${MODEL_NAME:-Qwen3-VL-8B-Instruct}"
export MODEL_PATH
export CONDA_NAME="${CONDA_NAME_EVAL}"
export USE_VLLM=False
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export ENABLE_PCA_VIZ=False
export ENABLE_CKNNA_VIZ=False

# ---- NeoCXR ----
export OUTPUT_PATH="${OUT_NEO}"
bash "${MEDEVALKIT_ROOT}/scripts/sbatch_eval_neocxr_qwen3vl_thinking_full_l40.sh"

export REPORT_METRICS_MODEL_RESULTS_ROOT="${OUT_NEO}/Qwen3-VL-8B-Instruct"
export REPORT_METRICS_DATASETS="NeoCXR"
bash "${MEDEVALKIT_ROOT}/sbatch_cal_report_metrics_neocxr_l40.sh"

# ---- NeoCXR-EV ----
export OUTPUT_PATH="${OUT_EV}"
bash "${MEDEVALKIT_ROOT}/scripts/sbatch_eval_neocxrev_qwen3vl_thinking_full_l40.sh"

export REPORT_METRICS_MODEL_RESULTS_ROOT="${OUT_EV}/Qwen3-VL-8B-Instruct"
export REPORT_METRICS_DATASETS="NeoCXR_EV"
bash "${MEDEVALKIT_ROOT}/sbatch_cal_report_metrics_neocxr_l40.sh"

echo "[DONE] Neo: ${OUT_NEO}"
echo "[DONE] EV:  ${OUT_EV}"
