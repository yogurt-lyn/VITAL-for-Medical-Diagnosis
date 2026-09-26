#!/usr/bin/env bash
# MIMIC-CXR 4k ablations on vis-sigmoid + gap×k3 (2362850 setup):
#   - gap k3 weight α ∈ {1, 4}  @ noise σ=0.10 (baseline α=2, σ=0.10)
#   - noise σ ∈ {0.05, 0.15, 0.20}  @ α=2  (5% / 15% / 20%)
#
#   bash scripts/submit_mimic_4k_gapkl_ablation.sh
#   RUN_ALPHA=0 bash scripts/submit_mimic_4k_gapkl_ablation.sh   # noise only (3 trains)
#   RUN_NOISE=0 bash scripts/submit_mimic_4k_gapkl_ablation.sh   # α only (2 trains)
#   PARALLEL=1 bash scripts/submit_mimic_4k_gapkl_ablation.sh

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MVP_ROOT="${MVP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs}"
DATA_DIR="${PROJECT_ROOT}/data/mimiccxr_verl_4k_imgcap1500"
PROJECT_NAME="${PROJECT_NAME:-mimiccxr_opd_8b2b_n4_tvis_sigmoid_gapkl}"
RUN_ALPHA="${RUN_ALPHA:-1}"
RUN_NOISE="${RUN_NOISE:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${MVP_ROOT}/slurm_logs"

if [[ ! -f "${DATA_DIR}/train_grpo_teacher_rougel.parquet" ]]; then
  echo "[FATAL] missing ${DATA_DIR}/train_grpo_teacher_rougel.parquet" >&2
  exit 1
fi

OPD_FLAGS=( )
if [[ -n "${A800_NODELIST:-}" ]]; then
  OPD_FLAGS+=( --nodelist="${A800_NODELIST}" )
  echo "[INFO] pin A800 to ${A800_NODELIST}"
fi

submit_one() {
  local label="$1"
  local experiment_name="$2"
  local tag="$3"
  local alpha="$4"
  local noise="$5"
  local dep="${6:-}"

  local ckpt_root="${MVP_ROOT}/checkpoints/${PROJECT_NAME}/${experiment_name}"
  local dep_arg=( )
  if [[ -n "${dep}" ]]; then
    dep_arg=( --dependency="afterok:${dep}" )
  fi

  local train_jid
  if ! train_jid=$(sbatch --parsable "${dep_arg[@]}" "${OPD_FLAGS[@]}" \
    --export=ALL,TRAIN_FILE="${DATA_DIR}/train_grpo_teacher_rougel.parquet",PROJECT_NAME="${PROJECT_NAME}",EXPERIMENT_NAME="${experiment_name}",ROUGEL_GAP_KL_COEF="${alpha}",TEACHER_VISUAL_NOISE_STD="${noise}" \
    scripts/sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh); then
    echo "[FATAL] sbatch failed for ${label} (MaxSubmitJobsPerAccount?)" >&2
    exit 1
  fi
  echo "[SUBMIT] ${label} train=${train_jid} alpha=${alpha} noise_std=${noise}" >&2

  bash scripts/submit_eval_three_ds_parallel.sh \
    --tag "${tag}" \
    --ckpt "${ckpt_root}" \
    --train-job "${train_jid}" \
    --afterok "${train_jid}"

  echo "${train_jid}"
}

PARALLEL="${PARALLEL:-0}"
LAST=""
J_A1="—"
J_A4="—"
J_N005="—"
J_N015="—"
J_N020="—"

run_with_dep() {
  local label="$1"
  local exp="$2"
  local tag="$3"
  local alpha="$4"
  local noise="$5"
  local dep=""
  if [[ "${PARALLEL}" != "1" && -n "${LAST}" ]]; then
    dep="${LAST}"
  fi
  submit_one "${label}" "${exp}" "${tag}" "${alpha}" "${noise}" "${dep}"
}

if [[ "${RUN_ALPHA}" == "1" ]]; then
  J_A1=$(run_with_dep \
    "abl_a1" \
    "qwen3vl2b_student_8bgspo_teacher_opd_n4_tvis_sigmoid_gapkl_a1_4k_imgcap1500" \
    "opd_gapkl_4k_a1" \
    "1.0" "0.10")
  LAST="${J_A1}"
  J_A4=$(run_with_dep \
    "abl_a4" \
    "qwen3vl2b_student_8bgspo_teacher_opd_n4_tvis_sigmoid_gapkl_a4_4k_imgcap1500" \
    "opd_gapkl_4k_a4" \
    "4.0" "0.10")
  LAST="${J_A4}"
fi

if [[ "${RUN_NOISE}" == "1" ]]; then
  J_N005=$(run_with_dep \
    "abl_n005" \
    "qwen3vl2b_student_8bgspo_teacher_opd_n4_tvis_sigmoid_gapkl2_noise005_4k_imgcap1500" \
    "opd_gapkl_4k_n005" \
    "2.0" "0.05")
  LAST="${J_N005}"
  J_N015=$(run_with_dep \
    "abl_n015" \
    "qwen3vl2b_student_8bgspo_teacher_opd_n4_tvis_sigmoid_gapkl2_noise015_4k_imgcap1500" \
    "opd_gapkl_4k_n015" \
    "2.0" "0.15")
  LAST="${J_N015}"
  J_N020=$(run_with_dep \
    "abl_n020" \
    "qwen3vl2b_student_8bgspo_teacher_opd_n4_tvis_sigmoid_gapkl2_noise020_4k_imgcap1500" \
    "opd_gapkl_4k_n020" \
    "2.0" "0.20")
  LAST="${J_N020}"
fi

cat <<EOF
[MIMIC 4k gap-kl ablations]
  data=${DATA_DIR}
  baseline: α=2, σ=0.10 (10%, job 2362850)

  abl_a1   train=${J_A1}   α=1.0  σ=0.10
  abl_a4   train=${J_A4}   α=4.0  σ=0.10
  abl_n005 train=${J_N005} α=2.0  σ=0.05 (5%)
  abl_n015 train=${J_N015} α=2.0  σ=0.15 (15%)
  abl_n020 train=${J_N020} α=2.0  σ=0.20 (20%)
  RUN_ALPHA=${RUN_ALPHA} RUN_NOISE=${RUN_NOISE} PARALLEL=${PARALLEL}
EOF
