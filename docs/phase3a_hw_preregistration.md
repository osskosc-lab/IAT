# IAT Phase 3A-HW v1.0 — Formal preregistration

**Status: PILOT_READY / hardware parameters not yet frozen.**

This protocol tests one bounded question: after the observable pre-probe state is matched, does the current trial's earlier stimulus order add out-of-device predictive information about a common probe response?

A software PASS is not hardware evidence. A later hardware PASS would support only the observable-only predictive statement defined here; it would not establish a new energy, a quantum effect, or a new law of nature.

## Primary intervention

Each device receives two histories containing the same A and B stimuli. Only order differs:

- `AB`: A -> B -> common matching interval -> common probe
- `BA`: B -> A -> common matching interval -> common probe

A/B content, duration, total dose, common-probe waveform, acquisition settings, and analysis window are fixed before the first hardware main trial. Order-specific matching times are forbidden.

## Fixed sample

Eight devices are used. Each contributes 120 main trials (60 AB, 60 BA) plus 24 long-washout negative-control trials, for 1,152 total trials. The main schedule has six blocks of 20 trials/device, exactly 10 AB and 10 BA per block, with maximum same-order run length three. The compressed schedule is frozen in `config/phase3a_hw_schedule.json`; its generated CSV digest is fixed in `config/phase3a_hw_preregistration.yaml`.

No outcome-dependent extension or early success stopping is allowed. Early stopping is limited to safety, device damage, acquisition failure, or data-storage failure.

## Observable state

No latent capacitor state, hidden simulator coordinate, or theory-only state is allowed in the primary analysis. The pre-probe observables are terminal voltage, terminal current, current fraction of full scale, terminal-voltage slope, and measured temperature deviation.

For the four preregistered equivalence coordinates, the complete 99% device-bootstrap CI of the standardized AB-BA difference must lie inside `[-0.10,+0.10]`.

## Outcome

The primary outcome is ordinary electrical probe work,

`W_P = integral voltage(t) * current(t) dt`

over the frozen common-probe window. Secondary outcomes may be stored, but the primary outcome cannot be replaced after outcome inspection.

## Models

`M0` is an observable-current-state model with acquisition block, run index, previous-trial order, and measured reset elapsed time as nuisance covariates. It uses the frozen r3-style standardized hinge-spline map and low-order interactions.

`MH` is exactly `M0` plus current-trial order. The primary statistic is held-out-device `RMSE(MH)/RMSE(M0)` using leave-one-device-out evaluation.

`M0*`/`MH*` repeat the test with a more flexible observable-only basis including cubic terms and frozen pairwise interactions.

## Inference

The device is the resampling unit. The frozen bootstrap uses 20,000 device resamples and a 99% interval. Primary support requires the 99% upper bound of `RMSE(MH)/RMSE(M0)` to be below `0.90`.

## Pre-hardware software calibration repair

Before any empirical hardware outcome existed, the first complete synthetic software-audit run exposed one precision problem in the negative controls. With the fixed 24 long-washout trials per device, a no-current-order generator gave a long-washout `MH/M0` 99% interval of approximately `[0.989, 1.062]`. The original `[0.95,1.05]` equivalence interval was therefore too narrow for the frozen pilot sample size, even under the software null.

The negative-control equivalence band was repaired **before hardware parameter freeze and before any empirical outcome existed** to `[0.90,1.10]`, matching the 10% minimum-effect scale of the primary support rule. The previous-trial carryover attack used in CI remains far outside this interval and must still fail G5. This repair is recorded as `r1-negative-control-margin` in the YAML preregistration; no empirical data may be used to revise it further.

## Gates

- **G0 Integrity:** exact frozen schedule, exact trial/device counts, unique IDs, finite values, per-block balance, and run-length rule.
- **G1 Current-state equivalence:** all four observable-state 99% intervals wholly inside `[-0.10,+0.10]`.
- **G2 Primary history value:** 99% upper bound of `MH/M0 < 0.90`.
- **G3 Cross-device consistency:** median device ratio `<0.90` and at least 7/8 device ratios `<1`.
- **G4 Temporal-order falsification:** 99% lower bound of conditionally shuffled/ordered MH RMSE `>1.10`.
- **G5 Previous-trial carryover:** previous-trial/reset-only diagnostic ratio 99% CI wholly inside `[0.90,1.10]`.
- **G6 Long-washout null:** long-washout `MH/M0` 99% CI wholly inside `[0.90,1.10]`.
- **G7 Observable-confound hardening:** 99% upper bound of `MH*/M0* <0.90`.

`PILOT_SUPPORT` is allowed only if G0-G7 all pass. Otherwise the first failing scientific explanation is retained as the final classification: implementation failure, current-state confound, primary null, device-specific result, order-mechanism failure, carryover artifact, hardware-null failure, or observable-model explanation.

## Independent validation

`src/phase3a_hw_validator.py` does not import the analysis module. From saved held-out prediction artifacts it independently recomputes hashes, RMSE ratios, device-bootstrap intervals, current-state equivalence intervals, device consistency, Gate states, and final classification.

## Before hardware acquisition

`config/phase3a_hw_hardware_manifest.yaml` must be completed and changed from `NOT_FROZEN` to `FROZEN` before the first hardware main trial. A/B/probe waveforms, matching time, long-washout time, measurement ranges, sampling rate, clipping rule, timestamp tolerance, temperature measurement, and safety limits must be fixed in that commit.

The current CI stage therefore performs only software falsification/regression tests: a no-current-order synthetic null, a clean current-order positive control, and a previous-trial carryover attack. These are implementation tests, not empirical IAT results.
