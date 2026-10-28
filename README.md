# SiteGuard-CP

Shift-aware conformal prediction for autism connectome classifiers at **unseen acquisition sites**.

A classifier trained on some hospitals is deployed at a new one. Does its conformal "90% coverage" guarantee still hold there, and can it be repaired using only **unlabeled** scans from the new site? This repository audits that question on ABIDE-I with leave-one-site-out evaluation and implements two label-free repairs. Everything runs on a CPU, including free Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sivahari1/SiteGuard-CP/blob/main/notebooks/SiteGuardCP_ABIDE_Colab.ipynb)

## Main results (ABIDE-I, 18 held-out sites, target coverage 0.90)

| Method | Worst-site coverage | Sites below 0.85 | Mean set size | Single-label calls |
|---|---|---|---|---|
| Mondrian (class-conditional) CP | 0.830 | 3 | 1.62 | 38% |
| **WM-CP, score-space weights** | 0.852 | **0** | **1.66** | 34% |
| **Leave-site-out margin** | 0.855 | **0** | 1.71 | 29% |
| WM-CP, PCA weights | 0.857 | 0 | 1.77 | 23% |
| Hybrid weights, EM prevalence | 0.715 | 4 | 1.68 | 32% |

- Both label-free repairs remove every site below 0.85 (paired Wilcoxon p ≤ 0.003 against Mondrian).
- At **equal set size**, PCA-space weighting loses to a stricter Mondrian (worst site 0.857 vs 0.893), so its gain is extra set size. Score-space weighting stays slightly ahead of Mondrian at every α tested, but not significantly (p = 0.17 to 0.57).
- Correcting for site differences in autism prevalence fails: unlabeled prevalence estimates are off by 0.22 (EM) to 0.29 (BBSE), and even the true prevalence adds nothing.
- Held-out sites are identifiable from their connectome alone (domain AUC 0.84 to 0.99, median 0.95).

All numbers come from `results/abide_aal_tangent_run2/` and can be regenerated with the steps below.

## Repository layout

```
notebooks/SiteGuardCP_ABIDE_Colab.ipynb   full pipeline: data, LOSO loop, all conformal methods, analyses, figures
siteguard/toolbox.py                      conformal toolbox (weighted / Mondrian quantiles, density ratios, tempering, EM)
scripts/paper_analysis.py                 equal-set-size tests, leave-site-out margin, paper figures, from the notebook cache
results/abide_aal_tangent_run2/           per-site results, summary tables and figures from the reported run
```

## How to run

**Colab (recommended).** Open the notebook with the badge above. Choose *Runtime > Change runtime type > CPU*, then *Run all*. The first run downloads about 150 MB of ABIDE-I ROI time series and takes roughly an hour on a free CPU runtime (18 leave-one-site-out folds × 5 calibration splits). Set `USE_SYNTHETIC=True` in the config cell for a one-minute smoke test without any download.

**Local.**

```bash
pip install -r requirements.txt
jupyter notebook notebooks/SiteGuardCP_ABIDE_Colab.ipynb
```

**Re-run the analyses without refitting.** The notebook saves `siteguard_results/siteguard_cache.pkl` (per-site predicted probabilities and domain features). From that cache:

```bash
python scripts/paper_analysis.py --results siteguard_results
```

This takes about a minute and writes the equal-set-size tests, the leave-site-out margin rows and the figures. Only load cache files produced by your own run.

## Configuration

All settings live in the `CFG` dictionary in the notebook's first cell. Defaults reproduce the reported run: C-PAC pipeline, band-pass filtered, no global signal regression, quality-checked subjects (871), AAL atlas (116 ROIs, 6,670 edges), tangent-space connectivity fit on training sites only, L2 logistic regression (C = 0.05), α = 0.10, 30% calibration split, 5 splits per site, ESS floor 0.20, weight clip 20.

## Data

ABIDE-I data are not included. The notebook downloads the Preprocessed Connectomes Project derivatives through `nilearn.datasets.fetch_abide_pcp`. Use of ABIDE data is subject to the ABIDE data usage terms; please acknowledge ABIDE and the Preprocessed Connectomes Project in any work that uses them. The `results/` folder contains only site-level aggregates, no per-subject data.

## Citation

A paper describing this work has been submitted to the International Workshop on AI in Mental Health (CARES 2027, NIT Goa). Citation details will be added on acceptance.

## License

Code: MIT (see `LICENSE`).
