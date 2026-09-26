# Data lists

Images and model weights are not in this repo. These JSONL files are the sample lists used by the MIMIC-4k / NeoCXR training splits and the eval sets.

Each line: `dataset`, `role`, `id`, `image`.

| file | n | role |
|---|---:|---|
| `data/splits/mimic4k_sft.jsonl` | 1000 | SFT only. First 1000 of the 4k stratified sample (seed 42). Not used as OPD rollouts. |
| `data/splits/mimic4k_rl_opd.jsonl` | 3000 | RL / OPD. The other 3000. VITAL trains on `train_grpo_teacher_rougel.parquet` built from this split. |
| `data/splits/neocxr_sft.jsonl` | 1054 | NeoCXR SFT (`train_sft.parquet`). |
| `data/splits/neocxr_rl_opd.jsonl` | 3165 | NeoCXR RL / OPD (`train_grpo.parquet`). |
| `data/splits/neocxr_test.jsonl` | 1434 | NeoCXR test. Eval only. |
| `data/splits/neocxr_ev_eval.jsonl` | 1065 | NeoCXR-EV external test. Eval only. No SFT and no RL split. |
| `data/splits/chexpert_rrg_impression_eval.jsonl` | 202 | CheXpert-plus-RRG impression section. Eval only. No SFT and no RL split in this project. |

MIMIC 4k construction (`scripts/preprocess_mimiccxr_verl.py`, seed 42):

- Draw 4000 train studies, stratified by CheXpert label signature to match the frontal test set.
- Of those, 1000 are SFT (`--sft_samples 1000`).
- The remaining 3000 are GRPO / OPD.
- OPD additionally needs a teacher ROUGE-L cache parquet (`train_grpo_teacher_rougel.parquet`). That file is not shipped; rebuild it with the teacher merged checkpoint on the RL split.

CheXpert and NeoCXR-EV are held-out eval sets. Do not train on them.
