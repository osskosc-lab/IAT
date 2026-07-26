# IAT — Phase 2C-r2A Confirmatory Experiment

This repository runs the next preregistered stage after **IAT Phase 2C-r1.2.1**.

## Scientific objective

The experiment separates two claims that must not be conflated:

1. **Structure-recovery claim:** an adaptive history model can recover unknown one- and two-lag structures, detect a lag outside the candidate range, avoid false history selection under a Markov null, and lose performance when temporal correspondence is destroyed.
2. **Dual-lag effect-size claim:** a second lag provides at least 5% extra predictive value over the best single-lag model.

Phase 2C-r2A tests claim 1 only. Claim 2 remains unsupported by r1.2.1 and requires a separate pilot before any new confirmatory run.

## Frozen design

- Generator: stable linear-Gaussian delay systems
- Candidate lags: 1–6
- Diagnostic lags: 7–10
- Adaptive structures: Markov + zero, one, or two state-history lags
- Strong baseline: DelayARX with state and input lags 1–6
- Train / inner / outer / diagnostic: 80 / 20 / 20 / 20 trajectories per seed
- Test: step, pulse, chirp × amplitudes 0.4 and 0.8; 40 trajectories per cell
- Confirmatory seeds: 30
- Inference: 99% seed-cluster bootstrap, 4,000 repetitions
- Finite-sample audit: 40 training trajectories

The estimator never receives the true lag set. The generator stores it only for final independent validation.

## Preregistered Gates

| Gate | Test | Criterion |
|---|---|---|
| 0 | implementation integrity | all checks pass |
| 1 | in-support exact recovery | 99% CI lower > 0.90 |
| 2 | Adaptive / DelayARX non-inferiority | 99% CI upper < 1.05 |
| 3 | out-of-support detection | 99% CI lower > 0.90 |
| 4 | Markov-null false-positive rate | 99% CI upper < 0.10 |
| 5 | temporal-order mechanism | shuffle / ordered 99% CI lower > 1.10 |
| 6a | finite-sample recovery | 99% CI lower > 0.80 |
| 6b | finite-sample non-inferiority | 99% CI upper < 1.05 |

## Automated execution

GitHub Actions performs:

```text
freeze check
→ unit tests
→ smoke experiment
→ 30-seed confirmatory experiment
→ independent validation
→ CSV / JSON / PNG / Markdown / PDF artifact upload
```

Manual execution:

```bash
python -m pip install -r requirements.txt
python src/check_freeze.py
pytest -q
python src/r2a_experiment.py --mode smoke
python src/r2a_experiment.py --mode confirm
python src/r2a_validator.py
```

## Outputs

The workflow artifact `IAT-Phase2C-r2A-results` contains:

- `metrics.csv`
- `selection.csv`
- `trajectory_hashes.csv`
- `truth_key.json`
- `validation_report.json`
- `gate_table.csv`
- `validated_rows.csv`
- `gate_estimates.png`
- `summary.md`
- `IAT_Phase2C_r2A_report.pdf`

## Scientific boundary

A PASS would support only a restricted operational statement about unknown-lag selection and diagnostic behavior in this frozen linear-Gaussian toy-model family. It would not establish:

- superiority over correctly specified standard delay models;
- generalization to nonlinear systems or real observations;
- a quantum-process connection;
- information as an independent physical substance;
- a new law of nature.
