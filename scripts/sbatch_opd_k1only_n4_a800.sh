#!/usr/bin/env bash
# Official-style OPD: 4 on-policy samples/prompt, k1-PG only.
#   no task GSPO, no group-norm, no little-teacher, no visual distill.
#
#   sbatch scripts/sbatch_opd_k1only_n4_a800.sh

#SBATCH --job-name=opd_k1only_n4
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
MVP_ROOT="${MVP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs}"
CONDA_SH="${CONDA_SH:?set CONDA_SH to your conda.sh path}"
CONDA_NAME="${CONDA_NAME:-verl}"
source "${CONDA_SH}"
conda activate "${CONDA_NAME}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1

CUPY_PREFIX="${CUPY_PREFIX:-${PROJECT_ROOT}/.deps/cupy-cuda12x-13.6.0}"
if [[ ! -d "${CUPY_PREFIX}/cupy" ]]; then
  echo "[FATAL] missing local cupy 13.6.0 at ${CUPY_PREFIX}" >&2
  exit 1
fi
export PYTHONPATH="${CUPY_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}"
python -c "import cupy, numpy; print(f'[INFO] cupy={cupy.__version__} from {cupy.__file__}; numpy={numpy.__version__}')"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.8 >/dev/null 2>&1 || module load cuda/12.6 >/dev/null 2>&1 || true
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
unset RAY_ADDRESS RAY_NAMESPACE RAY_RUNTIME_ENV_JSON
unset PYTORCH_CUDA_ALLOC_CONF

SHORT_TMP=""
for d in /tmp /dev/shm; do
  if mkdir -p "${d}/o${SLURM_JOB_ID:-manual}" 2>/dev/null; then
    SHORT_TMP="${d}/o${SLURM_JOB_ID:-manual}"
    break
  fi
done
if [[ -z "${SHORT_TMP}" ]]; then
  echo "[FATAL] cannot create short tmp under /tmp or /dev/shm" >&2
  exit 1
fi
CACHE_ROOT="${CACHE_ROOT:-${SHORT_TMP}/c}"
RAY_TMPDIR="${RAY_TMPDIR:-${SHORT_TMP}/r}"
export RL_ZMQ_IPC_DIR="${RL_ZMQ_IPC_DIR:-${SHORT_TMP}/z}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor}"
mkdir -p "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${RAY_TMPDIR}" "${RL_ZMQ_IPC_DIR}" "${MVP_ROOT}/slurm_logs" "${MVP_ROOT}/checkpoints"

export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR

export N_GPUS_TRAINING="${N_GPUS_TRAINING:-2}"
export N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-1}"
export N_GPUS_TEACHER="${N_GPUS_TEACHER:-1}"
export TEACHER_TP="${TEACHER_TP:-1}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.55}"
export TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.75}"

export PROMPTMRG_ROOT="${PROMPTMRG_ROOT:-/path/to/MedEvalKit/PromptMRG}"
export PROMPTMRG_BERT_PATH="${PROMPTMRG_BERT_PATH:-${PROMPTMRG_ROOT}/checkpoints/bert-base-uncased}"
export MIMICCXR_CHEXPERT_DEVICE="${MIMICCXR_CHEXPERT_DEVICE:-cpu}"
export MIMICCXR_REWARD_W_ROUGE1="${MIMICCXR_REWARD_W_ROUGE1:-0.0}"
export MIMICCXR_REWARD_W_ROUGEL="${MIMICCXR_REWARD_W_ROUGEL:-0.5}"
export MIMICCXR_REWARD_W_CHEXPERT="${MIMICCXR_REWARD_W_CHEXPERT:-0.5}"

export STUDENT_MODEL="${STUDENT_MODEL:-${MVP_ROOT}/checkpoints/qwen3vl2b_instruct_mimiccxr_fullft_w5_v2_1k_noimgcap/global_step_250_merged_hf}"
export TEACHER_MODEL="${TEACHER_MODEL:-${PROJECT_ROOT}/checkpoints/mimiccxr_gspo_eq_r05c05_kl02_w5v2/qwen3vl8b_instruct_mimiccxr_gspo_eq_r05c05_kl02_w5v2_3k_imgcap1500/global_step_1500/actor_merged_hf}"
export PROJECT_NAME="${PROJECT_NAME:-mimiccxr_opd_8b2b_eq_k1_only_n4}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl2b_student_8bgspo_teacher_opd_k1_only_n4_3k_imgcap1500}"
export USE_TASK_REWARDS="${USE_TASK_REWARDS:-False}"
export ENTROPY_COEFF="${ENTROPY_COEFF:-0.0}"
export ROLLOUT_N="${ROLLOUT_N:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export GROUP_NORM_SIZE="${GROUP_NORM_SIZE:-4}"
export GROUP_NORM_COEF="${GROUP_NORM_COEF:-0.0}"
export LITTLE_TEACHER_COEF="${LITTLE_TEACHER_COEF:-0.0}"
export VISUAL_DISTILL_ENABLED="${VISUAL_DISTILL_ENABLED:-False}"

echo "[INFO] ===== OPD k1-only n=4: 8B GSPO teacher / 2B SFT student ====="
echo "[INFO] host=$(hostname) job=${SLURM_JOB_ID:-manual} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] rollout_n=${ROLLOUT_N} ppo_mini_batch=${PPO_MINI_BATCH_SIZE}"
echo "[INFO] loss = OPD k1-PG only (use_task_rewards=${USE_TASK_REWARDS}, no vis/group/little)"
echo "[INFO] student=${STUDENT_MODEL}"
echo "[INFO] teacher=${TEACHER_MODEL}"

cd "${PROJECT_ROOT}"
bash scripts/run_opd_mimiccxr_qwen3vl2b_fsdp.sh "$@"
