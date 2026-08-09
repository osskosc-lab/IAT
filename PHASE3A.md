# IAT Phase 3A — observable-only hardware transition

Phase 3A is implemented as a separate preregistered pipeline and does not overwrite the archived Phase 2C-r2A experiment.

```text
1,000 synthetic null calibrations
→ 8-device hardware pilot
→ freeze stimulus/probe/state gates/transform
→ SHA-256 lock
→ 30 new-device confirmatory run
→ independent validation
```

Current status: **PILOT OPEN**.

The synthetic null calibration has been executed for 1,000 independent null datasets and passed:

- median `RMSE(MH)/RMSE(M0)`: **1.001015**
- 0.5%–99.5% range: **[0.998041, 1.003447]**
- point-ratio false positive rate below 0.90: **0/1000**

This is a software-pipeline calibration, not evidence from an RC circuit. A confirmatory scientific claim is impossible until the eight-device pilot freezes the hardware settings and a new set of 30 devices is measured.

Key files:

- `docs/phase3a_preregistration.md`
- `config/phase3a_spec.yaml`
- `src/phase3a_prepare.py`
- `src/phase3a_analyze.py`
- `src/phase3a_null_calibration.py`
- `src/phase3a_validator.py`
- `.github/workflows/phase3a.yml`
- `reports/Phase3A/null_calibration_1000.json`
