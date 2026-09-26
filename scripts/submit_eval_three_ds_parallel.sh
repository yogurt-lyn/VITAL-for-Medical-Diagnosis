#!/usr/bin/env bash
# Submit merge + 3 independent one-dataset evals (MIMIC / IU / CheXpert) on different L40s.
# Usage:
#   bash scripts/submit_eval_three_ds_parallel.sh \
#     --tag TAG --ckpt CKPT_ROOT --train-job TRAIN_JOB_ID \
#     [--afterok TRAIN_SLURM_ID] [--skip-merge] [--nodes NODE1,NODE2,NODE3,NODE4]
#
# NODE1 = merge, NODE2 = MIMIC, NODE3 = IU, NODE4 = CheXpert.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MVP_ROOT="${MVP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs}"
cd "${PROJECT_ROOT}"

TAG=""
CKPT_ROOT=""
TRAIN_JOB_ID=""
AFTEROK=""
NODES=""
SKIP_MERGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --ckpt) CKPT_ROOT="$2"; shift 2 ;;
    --train-job) TRAIN_JOB_ID="$2"; shift 2 ;;
    --afterok) AFTEROK="$2"; shift 2 ;;
    --skip-merge) SKIP_MERGE=1; shift ;;
    --nodes) NODES="$2"; shift 2 ;;
    *) echo "[FATAL] unknown arg $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${TAG}" || -z "${CKPT_ROOT}" || -z "${TRAIN_JOB_ID}" ]]; then
  echo "usage: $0 --tag TAG --ckpt CKPT_ROOT --train-job TRAIN_JOB_ID [--afterok ID] [--skip-merge] [--nodes n1,n2,n3,n4]" >&2
  exit 1
fi

IFS=',' read -r NODE_MERGE NODE_MIMIC NODE_IU NODE_CXP <<< "${NODES}"

merge_flags=( )
eval_dep_flags=( )
if [[ -n "${AFTEROK}" ]]; then
  merge_flags+=( --dependency="afterok:${AFTEROK}" )
fi
if [[ -n "${NODE_MERGE}" ]]; then
  merge_flags+=( --nodelist="${NODE_MERGE}" )
fi

sbatch_retry() {
  local out="" i
  for i in 1 2 3 4 5 6; do
    if out="$(sbatch --parsable "$@" 2>/tmp/sbatch_eval_err.$$)"; then
      printf '%s\n' "${out}"
      rm -f /tmp/sbatch_eval_err.$$
      return 0
    fi
    echo "[WARN] sbatch failed (try ${i}): $(tr '\n' ' ' < /tmp/sbatch_eval_err.$$)" >&2
    sleep $((i * 15))
  done
  echo "[FATAL] sbatch failed after retries: $(cat /tmp/sbatch_eval_err.$$)" >&2
  rm -f /tmp/sbatch_eval_err.$$
  return 1
}

if [[ "${SKIP_MERGE}" == "1" ]]; then
  echo "[OK] skip actor merge tag=${TAG} ckpt=${CKPT_ROOT}"
  if [[ -n "${AFTEROK}" ]]; then
    eval_dep_flags+=( --dependency="afterok:${AFTEROK}" )
  fi
else
  MERGE_JOB="$(sbatch_retry \
    "${merge_flags[@]}" \
    --job-name="merge_${TAG}" \
    --export="ALL,CKPT_ROOT=${CKPT_ROOT},TRAIN_JOB_ID=${TRAIN_JOB_ID},DELETE_FSDP_ACTOR=1,DELETE_OLD_ACTOR_STEPS=1" \
    scripts/sbatch_merge_fsdp_actor_l40.sh)"
  echo "[OK] merge job=${MERGE_JOB} tag=${TAG}"
  eval_dep_flags+=( --dependency="afterok:${MERGE_JOB}" )
fi

submit_eval() {
  local ds="$1" subdir="$2" gt="$3" out="$4" max_new="$5" batch="$6" jname="$7" node="$8"
  local extra=( )
  extra+=( "${eval_dep_flags[@]}" )
  extra+=( --job-name="${jname}" )
  if [[ -n "${node}" ]]; then
    extra+=( --nodelist="${node}" )
  fi
  sbatch_retry \
    "${extra[@]}" \
    --export="ALL,CKPT_ROOT=${CKPT_ROOT},EVAL_DATASET=${ds},DATA_SUBDIR=${subdir},GT_MODE=${gt},OUTPUT_PATH=${out},MAX_NEW_TOKENS=${max_new},MIMIC_EVAL_BATCH_SIZE=${batch},USE_VLLM=False" \
    scripts/sbatch_eval_one_ds_qwen3vl_l40.sh
}

MIMIC_OUT="${MVP_ROOT}/eval_results/mimiccxr_${TAG}_job${TRAIN_JOB_ID}"
IU_OUT="${MVP_ROOT}/eval_results/iuxray_${TAG}_job${TRAIN_JOB_ID}_full3419_max512"
CXP_OUT="${MVP_ROOT}/eval_results/chexpert_rrg_impression_${TAG}_job${TRAIN_JOB_ID}"
MIMIC_BSZ="${EVAL_MIMIC_BATCH_SIZE:-8}"
IU_BSZ="${EVAL_IU_BATCH_SIZE:-32}"
CXP_BSZ="${EVAL_CXP_BATCH_SIZE:-32}"

J_MIMIC="$(submit_eval MIMIC_CXRfrontal MIMIC_CXRfrontal conversations "${MIMIC_OUT}" 256 "${MIMIC_BSZ}" "e_${TAG}_mimic" "${NODE_MIMIC:-}")"
J_IU="$(submit_eval IU_XRAY IU_XRAY findings_impression "${IU_OUT}" 512 "${IU_BSZ}" "e_${TAG}_iu" "${NODE_IU:-}")"
J_CXP="$(submit_eval CheXpert_Plus CheXpert_Plus_RRG_impression findings_impression "${CXP_OUT}" 512 "${CXP_BSZ}" "e_${TAG}_cxp" "${NODE_CXP:-}")"

echo "[OK] MIMIC job=${J_MIMIC} out=${MIMIC_OUT} node=${NODE_MIMIC:-any}"
echo "[OK] IU    job=${J_IU} out=${IU_OUT} node=${NODE_IU:-any}"
echo "[OK] CXP   job=${J_CXP} out=${CXP_OUT} node=${NODE_CXP:-any}"
