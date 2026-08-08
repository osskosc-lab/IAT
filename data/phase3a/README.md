# Phase 3A data contract

No confirmatory hardware data are committed yet.

## Raw samples

Place long-form samples at `data/phase3a/raw_samples.csv` and run:

```bash
python src/phase3a_prepare.py --raw data/phase3a/raw_samples.csv
```

Required columns are defined in `config/phase3a_spec.yaml`. Each row is one time sample and each trial must include at least two `pre` and two `probe` samples.

## Trial summaries

`phase3a_prepare.py` writes `data/phase3a/trials.csv`, one row per trial. This is the only table consumed by the primary observable-only analysis. Internal capacitor voltages, hidden simulator states, and latent memory variables are prohibited.

## Strong baseline predictions

Optional strong-baseline predictions use:

```text
trial_id,device_id,model,prediction
```

in `data/phase3a/strong_baseline_predictions.csv`. They must be generated without device leakage and on exactly the same held-out device folds.
