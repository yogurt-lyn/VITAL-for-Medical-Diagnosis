#!/usr/bin/env bash
# MIMIC-CXR Online Policy Distillation (OPD):
#   Teacher: Qwen3-VL-8B GSPO eq (r05c05)
#   Student: Qwen3-VL-2B SFT w5 v2
# Fully-async FSDP2 + standalone teacher vLLM (verl Multi-Teacher OPD).
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MVP_ROOT="${MVP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/outputs}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH:-}"
unset ROCR_VISIBLE_DEVICES
export VLLM_USE_V1=1
export NCCL_CUMEM_ENABLE=0
export NCCL_CUMEM_HOST_ENABLE=0

STUDENT_MODEL="${STUDENT_MODEL:-${MVP_ROOT}/checkpoints/qwen3vl2b_instruct_mimiccxr_fullft_w5_v2_1k_noimgcap/global_step_250_merged_hf}"
TEACHER_MODEL="${TEACHER_MODEL:-${PROJECT_ROOT}/checkpoints/mimiccxr_gspo_eq_r05c05_kl02_w5v2/qwen3vl8b_instruct_mimiccxr_gspo_eq_r05c05_kl02_w5v2_3k_imgcap1500/global_step_1500/actor_merged_hf}"
TRAIN_FILE="${TRAIN_FILE:-${PROJECT_ROOT}/data/mimiccxr_verl_4k_imgcap1500/train_grpo.parquet}"
TEST_FILE="${TEST_FILE:-${PROJECT_ROOT}/data/mimiccxr_verl_4k_imgcap1500/val_grpo.parquet}"

N_GPUS_TRAINING="${N_GPUS_TRAINING:-2}"
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-2}"
N_GPUS_TEACHER="${N_GPUS_TEACHER:-2}"
TEACHER_TP="${TEACHER_TP:-2}"
GEN_TP="${GEN_TP:-1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
MAX_NUM_TOKENS="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 8))"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-0}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-1}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ACTOR_LR="${ACTOR_LR:-5e-7}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-1500}"
SAVE_FREQ="${SAVE_FREQ:-1500}"
TEST_FREQ="${TEST_FREQ:--1}"

PROJECT_NAME="${PROJECT_NAME:-mimiccxr_opd_8b2b_eq_k1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3vl2b_student_8bgspo_teacher_opd_k1_3k_imgcap1500}"
SAVE_DIR="${SAVE_DIR:-${MVP_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}"
mkdir -p "${SAVE_DIR}"

if [[ ! -f "${STUDENT_MODEL}/config.json" ]]; then
  echo "[FATAL] student missing: ${STUDENT_MODEL}" >&2
  exit 1
fi
if [[ ! -f "${TEACHER_MODEL}/config.json" ]]; then
  echo "[FATAL] teacher missing: ${TEACHER_MODEL}" >&2
  exit 1
fi

echo "[INFO] OPD student=${STUDENT_MODEL}"
echo "[INFO] OPD teacher=${TEACHER_MODEL}"
echo "[INFO] use_task_rewards=${USE_TASK_REWARDS:-False} (False = OPD k1 only, no GSPO task loss)"
echo "[INFO] rollout_n=${ROLLOUT_N} ppo_mini_batch=${PPO_MINI_BATCH_SIZE} gen_batch=${GEN_BATCH_SIZE}"
echo "[INFO] group_norm_coef=${GROUP_NORM_COEF:-0.0} little_teacher_coef=${LITTLE_TEACHER_COEF:-0.0}"
echo "[INFO] clinical_token_weight_coef=${CLINICAL_TOKEN_WEIGHT_COEF:-0.0}"
echo "[INFO] teacher_visual_token_enabled=${TEACHER_VISUAL_TOKEN_ENABLED:-False} base_opd_coef=${TEACHER_VISUAL_BASE_OPD_COEF:-1.0} visual_coef=${TEACHER_VISUAL_TOKEN_COEF:-1.0} gate=${TEACHER_VISUAL_TOKEN_GATE:-sigmoid} top_fraction=${TEACHER_VISUAL_TOKEN_TOP_FRACTION:-0.3} sigmoid_tau=${TEACHER_VISUAL_SIGMOID_TAU:-0.5} sigmoid_bias=${TEACHER_VISUAL_SIGMOID_BIAS:-0.0} noise_std=${TEACHER_VISUAL_NOISE_STD:-0.10}"
echo "[INFO] gt_visual_sim_enabled=${GT_VISUAL_SIM_ENABLED:-False} base_opd_coef=${GT_VISUAL_BASE_OPD_COEF:-1.0} coef=${GT_VISUAL_SIM_COEF:-1.0} top_fraction=${GT_VISUAL_SIM_TOP_FRACTION:-0.3}"
echo "[INFO] chexpert_gap_base_opd_coef=${CHEXPERT_GAP_BASE_OPD_COEF:-1.0} gap_coef=${CHEXPERT_GAP_OPD_COEF:-0.0} normalize=${CHEXPERT_GAP_NORMALIZE:-True}"
echo "[INFO] rougel_gap_base_opd_coef=${ROUGEL_GAP_BASE_OPD_COEF:-1.0} gap_coef=${ROUGEL_GAP_OPD_COEF:-0.0} kl_coef=${ROUGEL_GAP_KL_COEF:-0.0} kl_mode=${ROUGEL_GAP_KL_MODE:-k3} zero_loops=${ROUGEL_GAP_KL_ZERO_LOOPS:-True} normalize=${ROUGEL_GAP_NORMALIZE:-True} modulate_visual=${ROUGEL_GAP_MODULATE_VISUAL:-False} vis_gap_gate=${TEACHER_VISUAL_GAP_GATE:-none} vis_gap_gate_tau=${TEACHER_VISUAL_GAP_GATE_TAU:-0.05}"
echo "[INFO] teacher_token_kl_coef=${TEACHER_TOKEN_KL_COEF:-0.0} mode=${TEACHER_TOKEN_KL_MODE:-k3} zero_loops=${TEACHER_TOKEN_KL_ZERO_LOOPS:-True}"
echo "[INFO] gap_dynamic_sampling=${GAP_DYNAMIC_SAMPLING:-False} keep_k=${GAP_DYNAMIC_KEEP_K:-4}"
echo "[INFO] gt_sat_metric=${GT_SAT_METRIC:-none} base_opd_coef=${GT_SAT_BASE_OPD_COEF:-1.0} coef=${GT_SAT_OPD_COEF:-0.0} normalize=${GT_SAT_NORMALIZE:-True} teacher_factor=${GT_SAT_USE_TEACHER_FACTOR:-False}"
echo "[INFO] GPUs train=${N_GPUS_TRAINING} rollout=${N_GPUS_ROLLOUT} teacher=${N_GPUS_TEACHER} teacher_tp=${TEACHER_TP}"
echo "[INFO] SAVE_DIR=${SAVE_DIR}"

python3 -m verl.experimental.fully_async_policy.fully_async_main \
  --config-path=config \
  --config-name=fully_async_ppo_trainer \
  actor_rollout_ref.hybrid_engine=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.image_key=images \
  data.truncation=error \
  data.filter_overlong_prompts=True \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${STUDENT_MODEL}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL:-0.6}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_NUM_TOKENS}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=verl.checkpoint_engine.nccl_checkpoint_engine \
  distillation.enabled=True \
  distillation.teacher_key=data_source \
  distillation.n_gpus_per_node="${N_GPUS_TEACHER}" \
  distillation.nnodes=1 \
  distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL}" \
  distillation.teacher_models.teacher_model.inference.name=vllm \
  distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
  distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL:-0.7}" \
  distillation.teacher_models.teacher_model.inference.enforce_eager=True \
  distillation.teacher_models.teacher_model.inference.temperature=1.0 \
  distillation.teacher_models.teacher_model.inference.prompt_length="${MAX_PROMPT_LENGTH}" \
  distillation.teacher_models.teacher_model.inference.response_length="${MAX_RESPONSE_LENGTH}" \
  distillation.teacher_models.teacher_model.inference.max_model_len="${MAX_NUM_TOKENS}" \
  distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${MAX_NUM_TOKENS}" \
  distillation.teacher_models.teacher_model.inference.max_num_seqs="${TEACHER_MAX_NUM_SEQS:-${ROLLOUT_N}}" \
  distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=False \
  distillation.teacher_models.teacher_model.inference.enable_prefix_caching=False \
  +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.mm_processor_cache_gb=0 \
  distillation.distillation_loss.loss_mode=k1 \
  distillation.distillation_loss.topk=32 \
  distillation.distillation_loss.use_task_rewards="${USE_TASK_REWARDS:-False}" \
  distillation.distillation_loss.use_policy_gradient=True \
  distillation.distillation_loss.distillation_loss_coef=1.0 \
  distillation.distillation_loss.loss_max_clamp=10.0 \
  distillation.distillation_loss.log_prob_min_clamp=-10.0 \
  distillation.distillation_loss.visual_distill_enabled="${VISUAL_DISTILL_ENABLED:-False}" \
  distillation.distillation_loss.visual_distill_coef="${VISUAL_DISTILL_COEF:-0.0}" \
  distillation.distillation_loss.visual_distill_loss="${VISUAL_DISTILL_LOSS:-cosine}" \
  distillation.distillation_loss.group_norm_coef="${GROUP_NORM_COEF:-0.0}" \
  distillation.distillation_loss.group_norm_by_std="${GROUP_NORM_BY_STD:-True}" \
  distillation.distillation_loss.group_norm_size="${GROUP_NORM_SIZE:-${ROLLOUT_N}}" \
  distillation.distillation_loss.little_teacher_coef="${LITTLE_TEACHER_COEF:-0.0}" \
  distillation.distillation_loss.clinical_token_weight_coef="${CLINICAL_TOKEN_WEIGHT_COEF:-0.0}" \
  distillation.distillation_loss.clinical_token_weight_normalize="${CLINICAL_TOKEN_WEIGHT_NORMALIZE:-True}" \
  distillation.distillation_loss.teacher_visual_token_enabled="${TEACHER_VISUAL_TOKEN_ENABLED:-False}" \
  distillation.distillation_loss.teacher_visual_base_opd_coef="${TEACHER_VISUAL_BASE_OPD_COEF:-1.0}" \
  distillation.distillation_loss.teacher_visual_token_coef="${TEACHER_VISUAL_TOKEN_COEF:-1.0}" \
  distillation.distillation_loss.teacher_visual_token_top_fraction="${TEACHER_VISUAL_TOKEN_TOP_FRACTION:-0.3}" \
  distillation.distillation_loss.teacher_visual_token_gate="${TEACHER_VISUAL_TOKEN_GATE:-sigmoid}" \
  distillation.distillation_loss.teacher_visual_sigmoid_tau="${TEACHER_VISUAL_SIGMOID_TAU:-0.5}" \
  distillation.distillation_loss.teacher_visual_sigmoid_bias="${TEACHER_VISUAL_SIGMOID_BIAS:-0.0}" \
  distillation.distillation_loss.teacher_visual_token_normalize="${TEACHER_VISUAL_TOKEN_NORMALIZE:-True}" \
  distillation.distillation_loss.teacher_visual_noise_std="${TEACHER_VISUAL_NOISE_STD:-0.10}" \
  distillation.distillation_loss.teacher_visual_gap_gate="${TEACHER_VISUAL_GAP_GATE:-none}" \
  distillation.distillation_loss.teacher_visual_gap_gate_tau="${TEACHER_VISUAL_GAP_GATE_TAU:-0.05}" \
  distillation.distillation_loss.teacher_visual_gap_boost_coef="${TEACHER_VISUAL_GAP_BOOST_COEF:-0.5}" \
  distillation.distillation_loss.gt_visual_sim_enabled="${GT_VISUAL_SIM_ENABLED:-False}" \
  distillation.distillation_loss.gt_visual_base_opd_coef="${GT_VISUAL_BASE_OPD_COEF:-1.0}" \
  distillation.distillation_loss.gt_visual_sim_coef="${GT_VISUAL_SIM_COEF:-1.0}" \
  distillation.distillation_loss.gt_visual_sim_top_fraction="${GT_VISUAL_SIM_TOP_FRACTION:-0.3}" \
  distillation.distillation_loss.gt_visual_sim_normalize="${GT_VISUAL_SIM_NORMALIZE:-True}" \
  distillation.distillation_loss.chexpert_gap_base_opd_coef="${CHEXPERT_GAP_BASE_OPD_COEF:-1.0}" \
  distillation.distillation_loss.chexpert_gap_opd_coef="${CHEXPERT_GAP_OPD_COEF:-0.0}" \
  distillation.distillation_loss.chexpert_gap_normalize="${CHEXPERT_GAP_NORMALIZE:-True}" \
  distillation.distillation_loss.chexpert_gap_group_size="${CHEXPERT_GAP_GROUP_SIZE:-${ROLLOUT_N}}" \
  distillation.distillation_loss.rougel_gap_base_opd_coef="${ROUGEL_GAP_BASE_OPD_COEF:-1.0}" \
  distillation.distillation_loss.rougel_gap_opd_coef="${ROUGEL_GAP_OPD_COEF:-0.0}" \
  distillation.distillation_loss.rougel_gap_normalize="${ROUGEL_GAP_NORMALIZE:-True}" \
  distillation.distillation_loss.rougel_gap_group_size="${ROUGEL_GAP_GROUP_SIZE:-${ROLLOUT_N}}" \
  distillation.distillation_loss.rougel_gap_modulate_visual="${ROUGEL_GAP_MODULATE_VISUAL:-False}" \
  distillation.distillation_loss.rougel_gap_kl_coef="${ROUGEL_GAP_KL_COEF:-0.0}" \
  distillation.distillation_loss.rougel_gap_kl_mode="${ROUGEL_GAP_KL_MODE:-k3}" \
  distillation.distillation_loss.rougel_gap_kl_zero_loops="${ROUGEL_GAP_KL_ZERO_LOOPS:-True}" \
  distillation.distillation_loss.rougel_gap_kl_vis_gate="${ROUGEL_GAP_KL_VIS_GATE:-False}" \
  distillation.distillation_loss.teacher_token_kl_coef="${TEACHER_TOKEN_KL_COEF:-0.0}" \
  distillation.distillation_loss.teacher_token_kl_mode="${TEACHER_TOKEN_KL_MODE:-k3}" \
  distillation.distillation_loss.teacher_token_kl_zero_loops="${TEACHER_TOKEN_KL_ZERO_LOOPS:-True}" \
  distillation.distillation_loss.gap_dynamic_sampling="${GAP_DYNAMIC_SAMPLING:-False}" \
  distillation.distillation_loss.gap_dynamic_keep_k="${GAP_DYNAMIC_KEEP_K:-4}" \
  distillation.distillation_loss.gt_sat_metric="${GT_SAT_METRIC:-none}" \
  distillation.distillation_loss.gt_sat_base_opd_coef="${GT_SAT_BASE_OPD_COEF:-1.0}" \
  distillation.distillation_loss.gt_sat_opd_coef="${GT_SAT_OPD_COEF:-0.0}" \
  distillation.distillation_loss.gt_sat_normalize="${GT_SAT_NORMALIZE:-True}" \
  distillation.distillation_loss.gt_sat_group_size="${GT_SAT_GROUP_SIZE:-${ROLLOUT_N}}" \
  distillation.distillation_loss.gt_sat_use_teacher_factor="${GT_SAT_USE_TEACHER_FACTOR:-False}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  trainer.logger='["console"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
  trainer.val_before_train=False \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.total_epochs=2 \
  trainer.resume_mode=disable \
  trainer.max_actor_ckpt_to_keep=1 \
  trainer.default_local_dir="${SAVE_DIR}" \
  trainer.default_hdfs_dir=null \
  rollout.nnodes=1 \
  rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
  rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
  async_training.staleness_threshold=0.5 \
  async_training.partial_rollout=True \
  async_training.trigger_parameter_sync_step=4 \
  async_training.require_batches=1 \
  async_training.use_trainer_do_validate=False \
  +ray_kwargs.ray_init.address=local \
  +ray_kwargs.ray_init.include_dashboard=False \
  +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR:-/tmp/r${SLURM_JOB_ID:-manual}}" \
  "$@"
