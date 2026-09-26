#!/usr/bin/env bash
# Vis-only OPD, native off. Sigmoid gate on |Δlogp|, not top-30% hard mask.
#   s_t = |log p_T(clean) - log p_T(x+ε)|
#   u_t = σ((z_t - β)/τ), z-score within the response, then mean-norm E[w]=1
#   A = w_vis * (-k1)

#SBATCH --job-name=opd_n4_tvis_sig
#SBATCH --output=outputs/slurm_logs/%x_%j.out
#SBATCH --error=outputs/slurm_logs/%x_%j.err
#SBATCH --partition=A800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=320G
#SBATCH --gres=gpu:a800:4
#SBATCH --time=48:00:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/mimiccxr_verl_4k_imgcap1500/train_grpo.parquet}"
export PROJECT_NAME="${PROJECT_NAME:-mimiccxr_opd_8b2b_n4_tvis_sigmoid}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl2b_student_8bgspo_teacher_opd_n4_tvis_sigmoid_3k_imgcap1500}"

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
export ROUGEL_GAP_OPD_COEF=0.0
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
export TEACHER_VISUAL_NOISE_STD=0.10

echo "[INFO] vis-only sigmoid OPD, native OFF"
bash "${PROJECT_ROOT}/scripts/sbatch_opd_k1only_n4_a800.sh" "$@"
