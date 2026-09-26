# Reproduction notes

These notes describe the checked-in scripts, not a newly validated training environment. Run commands from the repository root. No training or model evaluation was performed as part of the README redesign.

Local filesystem defaults use anonymous `/path/to/…` placeholders. Replace them with your own paths or pass the corresponding environment variables / CLI arguments before running.

## Environment and required inputs

- **CPython 3.12:** the original README identifies this version for the compiled distillation loss. The loader in [`losses.py`](../verl/verl/trainer/distillation/losses.py) searches for `_bytecode_backup/distillation_losses.pyc`. That compiled file is included in the checked-in tree; use the matching CPython version. Installing upstream `verl` is not a substitute.
- **GPU runtime:** compatible CUDA, PyTorch, vLLM and FSDP; package dependencies and optional extras are declared in [`setup.py`](../verl/setup.py). The repository does not provide a locked VITAL environment. `python -m pip install -e './verl[vllm]'` is a package installation step only.
- **Conda:** explicitly set `CONDA_SH` to your activation script (for example, `/path/to/miniconda3/etc/profile.d/conda.sh`); `CONDA_NAME` defaults to `verl`.
- **CuPy:** the base training wrapper requires `${CUPY_PREFIX}/cupy` to exist. Its default is `.deps/cupy-cuda12x-13.6.0`; this directory is not included. Supply a compatible local installation and set `CUPY_PREFIX`.
- **Models:** merged Hugging Face directories for the 2B SFT student and 8B GSPO teacher. The paper describes cold-start SFT and teacher GSPO; this README does not claim a complete launch recipe for those stages.
- **Data:** authorized image access, reference reports, training/validation parquets, and cached teacher ROUGE-L scores. The default VITAL wrapper checks for `train_grpo_teacher_rougel.parquet`. The cache and a dedicated cache-building entry point are not supplied.
- **Default GPUs:** the top-level recipe sets 2 training + 1 rollout + 1 teacher GPU. Slurm directives assume an A800 partition; adapt them to your cluster. If needed, pass host exclusions with `sbatch --exclude=...` at submission time. Plain `bash` uses the same script without scheduling resources.
- **Other local paths:** inspect `PROMPTMRG_ROOT`, `PROMPTMRG_BERT_PATH` and any environment-specific defaults in the launchers before use. Configure CUDA library directories through your environment's `LD_LIBRARY_PATH`; evaluation wrappers preserve it and prepend the metrics environment's library directory.

The preprocessing CLI is [`scripts/preprocess_mimiccxr_verl.py`](../scripts/preprocess_mimiccxr_verl.py); its arguments include `--train_json`, `--val_json`, `--test_json`, `--output_dir`, `--total_samples`, `--sft_samples`, and `--seed`. It does not supply the missing teacher-score cache. See [data splits](DATA_SPLITS.md).

## Implementation map

| Concept | Implementation / configuration |
| :--- | :--- |
| Teacher visual sensitivity, normalization, gates | [`visual_token_weights.py`](../verl/verl/trainer/distillation/visual_token_weights.py) |
| Positive task gaps; normalization over positive gaps within each prompt | [`gap_weights.py`](../verl/verl/trainer/distillation/gap_weights.py) |
| VITAL defaults | [`sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh`](../scripts/sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh) |
| Shared runtime and GPU allocation | [`sbatch_opd_k1only_n4_a800.sh`](../scripts/sbatch_opd_k1only_n4_a800.sh) |
| Async FSDP2 training entry | [`run_opd_mimiccxr_qwen3vl2b_fsdp.sh`](../scripts/run_opd_mimiccxr_qwen3vl2b_fsdp.sh) |

```text
A_t = w_vis,t * (log π_T - log π_S)
L   = L_VAP + α * E[w_gap * k3]
k3  = exp(Δ) - Δ - 1,  Δ = log π_T - log π_S
```

`ROUGEL_GAP_MODULATE_VISUAL=False` and `ROUGEL_GAP_OPD_COEF=0` keep the task gap out of the visual advantage. `ROUGEL_GAP_KL_COEF=2.0` scales the separate TAD term. `ROUGEL_GAP_KL_ZERO_LOOPS=True` applies the script's additional repetition filter (the previous README specifies 8-gram repeats ≥ 4); this does not gate VAP. The complete loss path has not been validated with a GPU training run.

## Ablations

After the same prerequisites and model/data exports used in the README:

```bash
# TAD coefficient, with image noise unchanged.
ROUGEL_GAP_KL_COEF=1.0 bash scripts/sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh
ROUGEL_GAP_KL_COEF=4.0 bash scripts/sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh

# Image Gaussian-noise standard deviation, with alpha unchanged.
TEACHER_VISUAL_NOISE_STD=0.05 bash scripts/sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh
TEACHER_VISUAL_NOISE_STD=0.15 bash scripts/sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh
TEACHER_VISUAL_NOISE_STD=0.20 bash scripts/sbatch_opd_k1only_n4_tvis_sigmoid_gapkl_a800.sh

# Visual-only OPD; no teacher ROUGE-L cache is required for this variant.
TRAIN_FILE=/path/to/train_grpo.parquet bash scripts/sbatch_opd_k1only_n4_tvis_sigmoid_a800.sh

# Native OPD with the base script's default visual/task gates disabled.
TRAIN_FILE=/path/to/train_grpo.parquet bash scripts/sbatch_opd_k1only_n4_a800.sh
```

Use a fresh shell without stale gate overrides when switching ablations. [`submit_mimic_4k_gapkl_ablation.sh`](../scripts/submit_mimic_4k_gapkl_ablation.sh) contains the cluster-specific Slurm sweep.

## Evaluation

MedEvalKit is external and not vendored. A compatible checkout must contain the generation and metric scripts referenced by these wrappers, plus dataset files, CheXbert and RaTE weights. Set `MEDEVALKIT_ROOT`, `CONDA_SH`, `CONDA_NAME_EVAL` and `CONDA_NAME_METRICS` as appropriate; the adult wrappers also use `CONDA_ENVS_DIR`. No unverified download link is supplied.

- **MIMIC-CXR:** `sbatch_eval_mimic_hf_model_l40.sh` requires `MODEL_PATH` and `OUTPUT_PATH`. It generates reports and merges NLG/CheXbert results into `metrics_all.json`.
- **CheXpertPlus:** `sbatch_eval_chexpert_rrg_impression_l40.sh` requires `MODEL_PATH` and `OUTPUT_PATH`; set `PROCESSOR_SRC` to an existing compatible processor directory if processor files need copying. This evaluates the 202-row impression subset, not the entire dataset.
- **Neonatal:** `sbatch_eval_mimic_model_on_neocxr_ev_l40.sh` requires `MODEL_PATH` and `OUTPUT_TAG` and runs both NeoCXR and NeoCXR-EV. It sets the dataset names internally. Its default model label and metric subdirectories are `Qwen3-VL-8B-Instruct`, even when `MODEL_PATH` points to a 2B model; inspect this routing before modifying `MODEL_NAME`. Select the matching neonatal-trained checkpoint to reproduce the paper's neonatal setting.

All Slurm evaluation examples assume an L40 partition. Create `outputs/slurm_logs` before submission. The manuscript results in the README were transcribed, not recomputed; consult [metric definitions](METRICS.md).
