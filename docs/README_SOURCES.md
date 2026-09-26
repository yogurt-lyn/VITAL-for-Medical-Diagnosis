# README sources and asset provenance

## Scientific source

The README uses the supplied anonymous manuscript **VITAL: Visual and Task Advantage Learning for On-Policy Distillation in Medical Diagnosis**, marked **Under review as a conference paper at ICLR 2027**. The supplied PDF has 17 pages and figure placeholders; the accompanying LaTeX ZIP supplies the original figures. No acceptance, spotlight designation, public paper URL, authorship, model release, or demo is asserted.

- Motivation: §1 and Figure 1.
- Method: §4.1–4.4, Equations 5–14, Algorithm 1.
- Main numbers: Table 1 (adult) and Table 2 (neonatal), corroborated by §4.6.
- Mechanism: Figures 3, 4 and 6.
- Qualitative OOD examples: Figures 10 and 12.

## Figure mapping

| Repository asset | Supplied source | Treatment |
| :--- | :--- | :--- |
| `hero.svg` | Paper title and model setup | New editorial banner; no empirical imagery |
| `ood-results.svg` | Tables 1–2 | New chart of exact reported F1 values; zero-based common scale |
| `motivation.png` | `Fig/F1.png` | Original visual content; ancillary metadata removed |
| `framework.pdf` | `Fig/F2.pdf` | Original vector figure, unchanged |
| `framework.png` | `Fig/F2.pdf` | Rasterized at 2400 px for GitHub rendering |
| `training-dynamics.png` | `Fig/V1.png` | Original visual content; ancillary metadata removed |
| `visual-dependence.png` | `Fig/Vcase.png` | Original visual content; ancillary metadata removed |
| `task-advantage.png` | `Fig/TAD.png` | Original visual content; ancillary metadata removed; ROUGE-L analysis from Figure 6 |
| `case-chexpert.png` | `Fig/Case2.png` | Original visual content; ancillary metadata removed |
| `case-neocxr-ev.png` | `Fig/Case3.png` | Original visual content; ancillary metadata removed |

The unused conceptual framework and mechanism drafts were removed once the original assets became available. The full manuscript and LaTeX source are not redistributed here.

## Reporting conventions

1. **OOD gains are percentage points:** 40.47 − 35.87 = 4.60 on CheXpertPlus, and 49.97 − 47.18 = 2.79 on NeoCXR-EV. The comparator is the strongest competing *distillation* method, not the teacher or undistilled student.
2. **Average definition:** Avg. follows the paper: the unweighted mean of ROUGE-L, ROUGE-1, METEOR, RaTE, precision, recall, and F1. BLEU-1/2 are excluded.
3. **TAD figure labeling:** `Fig/TAD2.png` uses F1 axes while the manuscript's Figure 5 caption says ROUGE-L. The README uses the internally consistent ROUGE-L figure `Fig/TAD.png` (Figure 6) instead.
4. **Selected cases:** original examples illustrate reported behavior; they are not clinical validation or aggregate evidence. Visual-dependence highlights are not region-level attention maps.
5. **Evaluation coverage:** neonatal metric code can skip empty or unparseable outputs; metric-valid sample counts may differ across models. See `METRICS.md`.

## Repository grounding

Commands and variables were checked against the existing training/evaluation wrappers and bundled package metadata. Missing assets and runtime assumptions are documented in `REPRODUCIBILITY.md`. No GPU jobs were run. The top-level license remains unspecified; the existing `verl/LICENSE` applies to the bundled framework and is not presented as a blanket license for new paper assets.
