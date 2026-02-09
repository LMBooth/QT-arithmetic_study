# Analysis Pipeline

This folder is the self-contained analysis workflow for any BIDS export of the arithmetic study.

The intended pattern is:
- point scripts to a BIDS root with `--bids-root`
- write derived outputs into `analysis_pipeline/`
- keep conversion logic separate in `conversion_package/`

## Project Status

Current completion state:
- Stage 0 complete: canonical trial table builder is implemented (`build_trial_table.py`).
- Stage 1 complete: dataset QC summary is implemented (`stage1_qc_summary.py`).
- Stage 2 complete: signal preprocessing is implemented (`stage2_preprocess.py`).
- Stage 3 complete: trial epoching is implemented (`stage3_epoch_trials.py`).
- Stage 4 complete: unimodal feature extraction is implemented (`stage4_extract_features.py`).
- Stage 5 complete: unified ML table assembly + split manifest is implemented (`stage5_build_fused_table.py`).
- Stage 6 complete: classic ML battery + split-aware benchmarking is implemented (`stage6_train_classic_ml.py`).
- Stage 7 complete: reproducible config-driven pipeline entrypoint is implemented (`run_pipeline.py` + `config/pipeline.yaml`).

## Stage 0: Canonical Trial Table (Completed)

`build_trial_table.py` converts `sub-*/eeg/*_events.tsv` into one row per arithmetic trial.

It pairs:
- cue markers (for example `1_5_2_4`)
- outcome markers (for example `1.5-2.4 Correct`)

It writes:
- calculation window: `calc_start_s` to `calc_end_s`
- answer window: `answer_start_s` to `answer_end_s`
- `response_time_s`
- difficulty, outcome, tutorial/main labels
- dropped sample counts per window

Run from repo root:

```powershell
python .\analysis_pipeline\build_trial_table.py `
  --bids-root .\bids_arithmetic `
  --strict
```

Default outputs:
- `analysis_pipeline/reports/trial_table_<bids_root_name>.tsv`
- `analysis_pipeline/reports/trial_table_<bids_root_name>_summary.json`

Optional explicit outputs:

```powershell
python .\analysis_pipeline\build_trial_table.py `
  --bids-root D:\data\bids_arithmetic `
  --out D:\outputs\trial_table.tsv `
  --summary-json D:\outputs\trial_table_summary.json
```

## Stage Details

### Stage 1: Dataset Audit and QC Summary (Completed)

Goal:
- produce a machine-readable and human-readable data quality snapshot before cleaning.

Inputs:
- BIDS root
- trial table TSV

Required work:
- summarize subject coverage by modality (EEG, ECG, pupil)
- summarize event consistency and per-subject trial counts
- summarize dropped sample burden per trial and per subject
- summarize pupil confidence quality and missingness
- save plots/tables for pre-cleaning QA

Implemented script:
- `analysis_pipeline/stage1_qc_summary.py`

Default outputs:
- `analysis_pipeline/reports/qc_dataset_summary.json`
- `analysis_pipeline/reports/qc_subject_table.tsv`
- `analysis_pipeline/reports/figures/*`

Run:

```powershell
python .\analysis_pipeline\stage1_qc_summary.py `
  --bids-root .\bids_arithmetic `
  --trial-table .\analysis_pipeline\reports\trial_table_bids_arithmetic.tsv
```

Notes:
- `--strict` will fail on anomalies (for CI/release checks).
- ECG QC now includes beat detection and per-subject HR metrics (`hr_mean_bpm`, percentiles, and ECG quality flag).
- ECG QC also includes baseline-minute vs task-window HR comparison (`baseline_hr_mean_bpm`, `task_hr_mean_bpm`, and `task_minus_baseline_hr_mean_bpm`).
- Known current anomalies in this dataset include extra `started_arithmetic` markers for `sub-006` and `sub-008`.

### Stage 2: Signal Cleaning / Preprocessing (Completed)

Goal:
- generate cleaned streams ready for epoching and feature extraction.

Required work:
- EEG filtering, robust bad-channel handling, rereferencing, optional ICA artifact removal
- ECG cleaning suitable for beat detection
- pupil confidence filtering, blink interpolation, smoothing/resampling
- log all per-subject processing decisions

Implemented script:
- `analysis_pipeline/stage2_preprocess.py`

Current outputs:
- `analysis_pipeline/derivatives/cleaned/<subject>/*`
- `analysis_pipeline/reports/preprocess_log.tsv`
- `analysis_pipeline/reports/preprocess_summary.json`

Run:

```powershell
python .\analysis_pipeline\stage2_preprocess.py `
  --bids-root .\bids_arithmetic `
  --overwrite
```

### Stage 3: Epoching from Trial Table (Completed)

Goal:
- create consistent epochs using trial table windows (not raw marker parsing in each script).

Required work:
- implement calc-window epochs (`calc_start_s` to `calc_end_s`)
- optionally implement answer-window epochs (`answer_start_s` to `answer_end_s`)
- include dropout-aware trial exclusion policy
- output epoch metadata manifest with keep/drop reasons

Implemented script:
- `analysis_pipeline/stage3_epoch_trials.py`

Windowing policy (agreed v1):
- make windowing programmable from CLI/config; do not hardcode 6 seconds.
- primary benchmark windows:
- `calc_fixed`: fixed length from `calc_start_s` (default 6.0 s).
- `full_trial`: `calc_start_s` to `answer_end_s` (variable length).
- optional analysis window: `answer_only` from `answer_start_s` to `answer_end_s`.
- default ML benchmark should use one epoch per trial (no overlap) to avoid inflated sample counts.
- optional augmentation mode can create sliding sub-windows, but must keep all sub-windows from the same `trial_id` in the same CV fold.
- if augmentation is used, report both:
- trial-level metrics (aggregate sub-window predictions to parent trial),
- and sub-window-level metrics (secondary only).
- quality gating remains per epoch/sub-window with minimum usable coverage (default 80%).

Recommended Stage 3 parameters:
- `--window-mode {calc_fixed,full_trial,answer_only}`
- `--fixed-window-s 6.0`
- `--min-coverage 0.80`
- `--sliding-window-s <float>` and `--sliding-step-s <float>` (disabled by default)
- `--allow-overlap` (default false)
- `--drop-short-windows` (default true; protects ECG short-window HRV reliability)

Expected outputs:
- `analysis_pipeline/derivatives/epochs/<subject>/*`
- `analysis_pipeline/reports/epoch_manifest.tsv`
- `analysis_pipeline/reports/epoch_summary.json`

Run:

```powershell
python .\analysis_pipeline\stage3_epoch_trials.py `
  --bids-root .\bids_arithmetic `
  --window-mode calc_fixed `
  --overwrite
```

### Stage 4: Feature Extraction (Completed)

Goal:
- extract trial-level features per modality.

Required work:
- EEG spectral and time-domain features
- ECG heart-rate / short-window variability features
- pupil dilation, variability, and quality features
- preserve feature provenance (source window and preprocessing version)

Implemented script:
- `analysis_pipeline/stage4_extract_features.py`

Feature battery (agreed v1):

Extraction and naming rules:
- unit of analysis: one row per `trial_id` per window (`calc`, `answer`)
- windows come from trial table columns only (`calc_start_s`, `calc_end_s`, `answer_start_s`, `answer_end_s`)
- per-subject reference baseline: first valid 60 seconds before the first `started_arithmetic` marker
- feature naming: `<modality>_<window>_<feature_name>` (example: `eeg_calc_theta_rel_frontal`)
- quality gate: if usable coverage in a window is below 80%, mark window features `n/a` and set quality flag
- preserve provenance columns in all feature tables: `participant_id`, `trial_id`, `block`, `difficulty_range`, `response_accuracy`, `window`, `preproc_version`

EEG features (from cleaned EEG, Welch PSD + time-domain descriptors):
- band absolute power by ROI for delta (1-4), theta (4-8), alpha (8-13), beta (13-30), high-beta (30-40)
- band relative power by ROI for delta/theta/alpha/beta/high-beta
- ROI band ratios: theta/alpha, theta/beta, alpha/beta
- frontal alpha asymmetry (log alpha power F4 minus F3)
- frontal-midline theta (Fz theta absolute and relative power)
- broadband Hjorth activity, mobility, complexity by ROI
- broadband variance, RMS, and line length by ROI
- spectral entropy by ROI
- baseline-normalized versions for key powers (theta, alpha, beta) as delta from subject baseline

EEG ROIs:
- frontal: Fp1, Fp2, F3, F4, F7, F8, Fz
- central: C3, C4, Cz
- parietal: P3, P4, P7, P8, Pz
- temporal: T7, T8
- occipital: O1, O2
- global: all EEG channels after preprocessing

ECG features (from cleaned ECG peak series, short-window safe):
- detected peak count in window
- valid RR interval count and RR coverage ratio
- HR mean, median, std, min, max (bpm)
- RR mean, SDNN, RMSSD, IQR (ms)
- pNN50 (%)
- coefficient of variation of RR
- baseline deltas: HR mean minus baseline HR mean, RMSSD minus baseline RMSSD
- ECG quality flag for insufficient beats (for example <3 peaks in window)

Pupil features (from cleaned pupil time series):
- pupil mean, median, std, IQR, min, max
- pupil 10th and 90th percentiles
- linear slope across window
- peak dilation (max minus early-window reference)
- area under curve above baseline
- first-derivative metrics: mean absolute velocity, max velocity
- confidence mean and low-confidence ratio
- valid sample coverage ratio
- gaze summary: x mean/std, y mean/std, gaze path length
- baseline-normalized pupil mean and peak dilation deltas

Feature sets to export:
- unimodal tables: `features_eeg.tsv`, `features_ecg.tsv`, `features_pupil.tsv`
- window variants: calc-only, answer-only, calc+answer concatenated
- fused table assembled later in Stage 5 from unimodal outputs

Current outputs:
- `analysis_pipeline/features/features_eeg.tsv`
- `analysis_pipeline/features/features_ecg.tsv`
- `analysis_pipeline/features/features_pupil.tsv`
- `analysis_pipeline/reports/feature_summary.json`

Notes:
- current pupil features include extreme outliers in this dataset (`pupil_mean_percentiles` in `feature_summary.json` show large upper-tail values), so Stage 5 should apply robust scaling and/or winsorization before model training.

Run:

```powershell
python .\analysis_pipeline\stage4_extract_features.py `
  --bids-root .\bids_arithmetic
```

### Stage 5: Unified ML Table Assembly

Goal:
- produce leakage-safe ML-ready tables for unimodal and fused comparisons with participant-aware split planning.

Implemented script:
- `analysis_pipeline/stage5_build_fused_table.py`

Implemented behavior:
- target label is difficulty bin (`difficulty_range` -> `target_label` / `difficulty_bin_index`)
- tutorial trials are configurable (`--include-tutorial`; default excludes tutorial)
- dropout gating is configurable (`--dropout-policy {none,absolute,subject_percentile}`)
- default dropout policy is absolute threshold (`--dropout-threshold 35`)
- multimodal fusion aligns epochs by `epoch_id` (safe for segmented/subwindow runs), not just `trial_id`
- unimodal outputs are always exported for modality-vs-fused benchmarking
- split manifest includes:
- leave-one-subject-out (LOSO) plans
- grouped participant holdout plans (default 10% and 20%, repeated)
- within-participant CV eligibility/recommended split counts

Current outputs:
- `analysis_pipeline/features/features_ml_eeg.tsv`
- `analysis_pipeline/features/features_ml_ecg.tsv`
- `analysis_pipeline/features/features_ml_pupil.tsv`
- `analysis_pipeline/features/features_fused.tsv`
- `analysis_pipeline/features/split_manifest.json`
- `analysis_pipeline/reports/fusion_summary.json`

Run (recommended defaults: main trials only, absolute dropout threshold 35):

```powershell
python .\analysis_pipeline\stage5_build_fused_table.py `
  --bids-root .\bids_arithmetic
```

Example variant (include tutorial + subject-relative dropout policy):

```powershell
python .\analysis_pipeline\stage5_build_fused_table.py `
  --bids-root .\bids_arithmetic `
  --include-tutorial `
  --unimodal-tag tutorial_subjectpct95 `
  --dropout-policy subject_percentile `
  --dropout-percentile 95
```

## Reproducible Run Chain (Stage 0-6)

From repo root:

```powershell
python .\analysis_pipeline\build_trial_table.py `
  --bids-root .\bids_arithmetic

python .\analysis_pipeline\stage1_qc_summary.py `
  --bids-root .\bids_arithmetic `
  --trial-table .\analysis_pipeline\reports\trial_table_bids_arithmetic.tsv

python .\analysis_pipeline\stage2_preprocess.py `
  --bids-root .\bids_arithmetic `
  --overwrite

python .\analysis_pipeline\stage3_epoch_trials.py `
  --bids-root .\bids_arithmetic `
  --window-mode calc_fixed `
  --overwrite

python .\analysis_pipeline\stage4_extract_features.py `
  --bids-root .\bids_arithmetic

python .\analysis_pipeline\stage5_build_fused_table.py `
  --bids-root .\bids_arithmetic

python .\analysis_pipeline\stage6_train_classic_ml.py `
  --bids-root .\bids_arithmetic
```

Stage 5 writes machine-readable manifests (`split_manifest.json`, `fusion_summary.json`) so the exact dataset selection and split policy are recorded with each run.

For tutorial-inclusive variants (used for baseline-proxy class scenarios in Stage 6), run Stage 5 with explicit outputs:

```powershell
python .\analysis_pipeline\stage5_build_fused_table.py `
  --bids-root .\bids_arithmetic `
  --include-tutorial `
  --dropout-policy absolute `
  --dropout-threshold 35 `
  --unimodal-tag tutorial_baseline `
  --fused-out .\analysis_pipeline\features\features_fused_tutorial_baseline.tsv `
  --split-manifest-out .\analysis_pipeline\features\split_manifest_tutorial_baseline.json `
  --summary-json .\analysis_pipeline\reports\fusion_summary_tutorial_baseline.json
```

### Stage 6: Classic ML Battery

Implemented script:
- `analysis_pipeline/stage6_train_classic_ml.py`

Implemented behavior:
- evaluates modalities independently (`eeg`, `ecg`, `pupil`) and combined (`fused`) from `split_manifest.json`
- supports LOSO, grouped participant holdout, and within-participant CV protocols
- runs nested tuning with leakage-safe preprocessing pipeline (impute -> quantile clip -> robust scale -> variance filter -> model)
- supports class-scenario experiments for omitting and merging classes (`--class-drop-labels`, `--class-merge`, `--class-merge-json`)
- supports optional tutorial-as-baseline proxy labeling (`--baseline-from-tutorial-label baseline`)
- supports optional deep tabular baselines when PyTorch is available (`cnn1d`, `transformer`)
- reports per-split and aggregate metrics: accuracy, balanced accuracy, macro-F1, weighted-F1, confusion matrices
- records captured fit/tuning warnings in `ml_results.json` and summary markdown (without flooding console logs)
- handles fit failures defensively per split/model/param set so one model failure does not abort the full run

Current outputs:
- `analysis_pipeline/models/*`
- `analysis_pipeline/reports/ml_results.json`
- `analysis_pipeline/reports/ml_summary.md`

Run (full defaults over all datasets/protocols/models):

```powershell
python .\analysis_pipeline\stage6_train_classic_ml.py `
  --bids-root .\bids_arithmetic
```

Run (faster validation profile):

```powershell
python .\analysis_pipeline\stage6_train_classic_ml.py `
  --bids-root .\bids_arithmetic `
  --max-outer-splits-per-protocol 2 `
  --max-param-combos 3 `
  --inner-folds 3 `
  --run-tag validation_small `
  --results-json .\analysis_pipeline\reports\ml_results_validation_small.json `
  --summary-md .\analysis_pipeline\reports\ml_summary_validation_small.md
```

Class scenario example (omit one bin and merge into 3 bins):

```powershell
python .\analysis_pipeline\stage6_train_classic_ml.py `
  --bids-root .\bids_arithmetic `
  --class-scenario-name three_level `
  --class-drop-labels 0.6-1.5 `
  --class-merge 1.5-2.4->low `
  --class-merge 2.4-3.3->low `
  --class-merge 3.3-4.2->mid `
  --class-merge 4.2-5.1->mid `
  --class-merge 5.1-6.0->high `
  --class-merge 6.0-6.9->high
```

Tutorial-as-baseline proxy example (baseline vs task):

```powershell
python .\analysis_pipeline\stage6_train_classic_ml.py `
  --bids-root .\bids_arithmetic `
  --split-manifest .\analysis_pipeline\features\split_manifest_tutorial_baseline.json `
  --datasets fused `
  --baseline-from-tutorial-label baseline `
  --class-scenario-name baseline_vs_task `
  --class-merge 0.6-1.5->task `
  --class-merge 1.5-2.4->task `
  --class-merge 2.4-3.3->task `
  --class-merge 3.3-4.2->task `
  --class-merge 4.2-5.1->task `
  --class-merge 5.1-6.0->task `
  --class-merge 6.0-6.9->task
```

Note: this baseline class is a proxy built from tutorial trials, not a true pre-task resting baseline epoch.

Post-hoc confusion highlights (top performers only):

```powershell
python .\analysis_pipeline\stage6_highlight_confusions.py `
  --results-json .\analysis_pipeline\reports\ml_results.json `
  --top-k-per-protocol 1 `
  --metric balanced_accuracy_mean `
  --out-json .\analysis_pipeline\reports\confusion_highlights.json `
  --out-md .\analysis_pipeline\reports\confusion_highlights.md
```

By default, confusion highlights place `baseline` next to the easiest class (lower-difficulty side). Use `--label-order-strategy as_is` to disable this.

Deep-model quick check (PyTorch required):

```powershell
python .\analysis_pipeline\stage6_train_classic_ml.py `
  --bids-root .\bids_arithmetic `
  --split-manifest .\analysis_pipeline\features\split_manifest_tutorial_baseline.json `
  --datasets fused `
  --protocols loso `
  --models cnn1d transformer `
  --max-outer-splits-per-protocol 1 `
  --max-param-combos 1 `
  --inner-folds 2 `
  --baseline-from-tutorial-label baseline `
  --class-scenario-name baseline_plus_grouped_4class_deep_quick `
  --class-drop-labels 6.0-6.9 `
  --class-merge 0.6-1.5->low_1_2 `
  --class-merge 1.5-2.4->low_1_2 `
  --class-merge 2.4-3.3->mid_3_4 `
  --class-merge 3.3-4.2->mid_3_4 `
  --class-merge 4.2-5.1->high_5_6 `
  --class-merge 5.1-6.0->high_5_6
```

Plots and table assets from Stage 6 outputs:

```powershell
python .\analysis_pipeline\stage6_build_report_assets.py `
  --run-manifest .\analysis_pipeline\reports\run_manifest_class_variants.json `
  --dataset-for-plots fused `
  --out-dir .\analysis_pipeline\reports\figures\stage6_class_variant_report
```

This writes:
- comparative metric plots (`plot_best_balanced_accuracy.png`, `plot_best_macro_f1.png`, `plot_balanced_accuracy_heatmap.png`, `plot_model_wins.png`)
- per-scenario confusion panel figures (`confusion_panel_*.png`)
- machine-readable + markdown tables (`table_*.csv`, `table_*.md`)

### Stage 7: Reproducibility and Packaging

Implemented files:
- `analysis_pipeline/config/pipeline.yaml`
- `analysis_pipeline/run_pipeline.py`
- `analysis_pipeline/reports/run_manifest.json`

Implemented behavior:
- one YAML config controls stage toggles, stage arguments, Stage 6 class scenarios, and confusion-highlights generation
- one command entrypoint executes selected stages in order and logs exact commands used
- run manifest captures start/end timestamps, command lines, return codes, and log file paths per step
- dry-run mode prints planned commands without execution

Run:

```powershell
python .\analysis_pipeline\run_pipeline.py `
  --config .\analysis_pipeline\config\pipeline.yaml
```

Run only Stage 6 for selected scenarios:

```powershell
python .\analysis_pipeline\run_pipeline.py `
  --config .\analysis_pipeline\config\pipeline.yaml `
  --only stage6 stage6_confusions `
  --stage6-scenarios all_bins three_level_merged
```

Configured class-variant run (requested comparison set):

```powershell
python .\analysis_pipeline\run_pipeline.py `
  --config .\analysis_pipeline\config\pipeline_class_variants.yaml `
  --only stage6 stage6_confusions
```

## Immediate Next Step for the Next Agent

Run full uncapped Stage 6 benchmarks per class scenario (or add new scenarios) and compare modality-vs-fused behavior under the same split protocols.
