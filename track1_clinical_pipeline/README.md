# Track A Clinical Pipeline

Production-style pipeline for diabetic complication prediction from tabular
clinical data.

Default study scope is now the paper-quality primary target set, with `DN`
as the active endpoint. Exploratory targets remain configurable in
`config.yaml`, but the default run is intentionally scoped to the clinically
defensible target.

## Run in PyCharm (recommended)

1. Open this folder as project root:
   `track1_clinical_pipeline`
2. Set Project Interpreter to Python 3.10+ virtualenv.
3. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run configuration:
   - `00_run_all_track_a`

This executes all stages in sequence:
1. preprocess
2. train_xgb
3. train_nn
4. evaluate

## Manual CLI run

```bash
python run_all_track_a.py --config config.yaml
```

## Outputs

- Processed artifacts: `processed/`
- Models and reports: `outputs/`
- Final review report: `outputs/audit_report.md`
- Calibrated thresholds: `outputs/decision_thresholds.json`

## Research-grade validation

The strengthened study plan is implemented as an additional fold-safe runner.
It starts from the raw CSV and fits preprocessing inside each fold, so it does
not reuse the single-split processed matrices.

Fast smoke test:

```bash
python src/research/clinical_cv.py --config config.yaml --preset quick
```

Main paper-ready model comparison:

```bash
python src/research/clinical_cv.py --config config.yaml --preset core --bootstrap 100
```

Feature-group ablations after the core run is stable:

```bash
python src/research/clinical_cv.py --config config.yaml --preset ablations --bootstrap 100
```

The old `--run-ablations` flag is still accepted, but it now maps to the
lighter `ablations` preset instead of running every model on every feature
group.

Important on Windows/conda: run these commands from an activated `(medai)`
terminal. Do not invoke `D:\Softwares\Miniconda\envs\medai\python.exe`
directly, because conda DLL paths may not be loaded.

Research outputs are written to `outputs/research_cv/`:

- `fold_metrics.csv`
- `fold_predictions.csv`
- `cv_summary.csv`
- `oof_summary.csv`
- `model_comparisons.csv`
- `leakage_audit.csv`
- calibration and decision-curve plots
- `research_validation_report.md`
