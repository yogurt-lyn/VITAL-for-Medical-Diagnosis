#!/usr/bin/env bash
# Vis-sigmoid OPD unchanged, plus a separate gap-weighted teacher KL:
#   A = w_vis * (-k1)
#   L += α * E[w_gap * k3(π_S, π_T)], α=2
#   w_gap = mean-norm max(m_T-m_S,0); copy-loops zeroed then re-normed.
# Native off. Does not mix gap into vis advantages.

#SBATCH --job-name=opd_n4_tvis_sig_gapkl
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=320G
#SBATCH --gres=gpu:a800:4
#SBATCH --time=48:00:00
#SBATCH --exclude=gpu8004

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/mimiccxr_verl_4k_imgcap1500/train_grpo_teacher_rougel.parquet}"
if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[FATAL] teacher parquet missing: ${TRAIN_FILE}" >&2
  exit 1
fi

export PROJECT_NAME="${PROJECT_NAME:-mimiccxr_opd_8b2b_n4_tvis_sigmoid_gapkl}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl2b_student_8bgspo_teacher_opd_n4_tvis_sigmoid_gapkl2_3k_imgcap1500}"

export USE_TASK_REWARDS=False
export ENTROPY_COEFF=0.0
export ROLLOUT_N=4
export PPO_MINI_BATCH_SIZE=4
export GROUP_NORM_SIZE=4
export GROUP_NORM_COEF=0.0
export LITTLE_TEACHER_COEF=0.0
export CLINICAL_TOKEN_WEIGHT_COEF=0.0
export VISUAL_DISTILL_ENABLED=False
export GT_VISUAL_SIM_ENABLED=False
export CHEXPERT_GAP_OPD_COEF=0.0
export GAP_DYNAMIC_SAMPLING=False
export GT_SAT_METRIC=none
export GT_SAT_OPD_COEF=0.0

export TEACHER_VISUAL_TOKEN_ENABLED=True
export TEACHER_VISUAL_BASE_OPD_COEF=0.0
export TEACHER_VISUAL_TOKEN_COEF=1.0
export TEACHER_VISUAL_TOKEN_GATE=sigmoid
export TEACHER_VISUAL_SIGMOID_TAU=0.5
export TEACHER_VISUAL_SIGMOID_BIAS=0.0
export TEACHER_VISUAL_TOKEN_NORMALIZE=True
export TEACHER_VISUAL_NOISE_STD="${TEACHER_VISUAL_NOISE_STD:-0.10}"

export ROUGEL_GAP_BASE_OPD_COEF=1.0
export ROUGEL_GAP_OPD_COEF=0.0
export ROUGEL_GAP_NORMALIZE=True
export ROUGEL_GAP_MODULATE_VISUAL=False
export ROUGEL_GAP_GROUP_SIZE=4
export ROUGEL_GAP_KL_COEF="${ROUGEL_GAP_KL_COEF:-2.0}"
export ROUGEL_GAP_KL_MODE=k3
export ROUGEL_GAP_KL_ZERO_LOOPS=True

echo "[INFO] vis-sigmoid + gap-weighted k3 KL coef=${ROUGEL_GAP_KL_COEF}, noise_std=${TEACHER_VISUAL_NOISE_STD}, loops zeroed, native OFF"
bash "${PROJECT_ROOT}/scripts/sbatch_opd_k1only_n4_a800.sh" "$@"
