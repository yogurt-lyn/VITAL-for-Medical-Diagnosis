# How metrics are computed

The paper defines **Avg.** as the unweighted mean of seven metrics, all expressed on the same percentage scale:

`Avg. = mean(RL, R1, MET, RaTE, P, R, F1)`

These are ROUGE-L, ROUGE-1, METEOR, RaTEScore, clinical precision, clinical recall, and clinical F1. BLEU-1, BLEU-2, and ROUGE-2 are not included in Avg. Evaluation scripts may also output these additional metrics; the paper's seven-metric definition governs the reported average.

## MIMIC-CXR frontal and CheXpert-plus-RRG impression

Greedy HF generation (temperature 0), then:

1. NLG vs the reference report (`conversations` gpt value on MIMIC; impression text on CheXpert):
   - ROUGE-L (`RL`), ROUGE-1 (`R1`)
   - BLEU-1, BLEU-2
   - METEOR
   - RaTEScore
   Script: MedEvalKit `utils/Metrics_Compute/cal_report_metrics_conversations_gt.py`
   Entry: `scripts/sbatch_eval_mimic_hf_model_l40.sh`, `scripts/sbatch_eval_chexpert_rrg_impression_l40.sh`
2. Clinical efficacy with CheXbert 14-label micro precision / recall / F1:
   - `P` = `ce_precision`, `R` = `ce_recall`, `F1` = `ce_f1`
   Script: MedEvalKit `PromptMRG/scripts/compute_chexbert_ce.py`
   Merged file: `metrics_all.json`

Canonical MIMIC eval size is the frontal test set (3398 scored when no responses are skipped). CheXpert impression eval is the 202-row list in `data/splits/chexpert_rrg_impression_eval.jsonl`.

## NeoCXR and NeoCXR-EV

Same NLG metrics, plus diagnosis micro P/R/F1 from the `Disease diagnosis:` span (not CheXbert).

- `P/R/F1` = `diagnosis_precision_micro` / `diagnosis_recall_micro` / `diagnosis_f1_micro` in `metrics_clinic.json`
- Dataset counts in paper Table 6 and `DATA_SPLITS.md` refer to split-list entries. If an evaluator skips empty or unparseable outputs, report its effective scored count separately; it does not replace the dataset size.
- Script: MedEvalKit `utils/Metrics_Compute/cal_report_metrics_NeoCXR.py`
- Entry: `scripts/sbatch_eval_mimic_model_on_neocxr_ev_l40.sh` (runs both NeoCXR and NeoCXR-EV; the wrapper sets `REPORT_METRICS_DATASETS` internally for each stage)

MedEvalKit itself is not vendored. Set `MEDEVALKIT_ROOT` to a checkout that contains those metric scripts and the CheXbert / RaTE weights.
