# IAT Phase 3A — Adversarial Falsification r1

## Status

Synthetic software falsification before the eight-device hardware pilot. This document is **not RC-circuit evidence**.

## Why this round was necessary

The first Phase 3A scaffold passed a simple no-order-effect null calibration, but that test did not challenge two important alternative explanations:

1. a weak linear current-state baseline can be beaten by order labels when the true response is nonlinear in observable pre-state variables;
2. laboratory drift can be mistaken for within-trial order history if stimulus orders are run in blocks or otherwise confounded with acquisition time.

Both are ordinary system-identification failures and must be ruled out before any hardware result can be interpreted as history-specific predictive value.

## Falsification experiment A — nonlinear observable-state confound

Generator: there is **no true order effect**. Order is correlated with terminal pre-voltage values that all remain within the provisional ±1 mV gate. The probe outcome depends quadratically on that observable pre-voltage.

Legacy analysis: linear M0 versus M0 + history order features.

Across 50 independent synthetic datasets:

- median MH/M0 = **0.4050**
- 5–95% range = **0.3911–0.4268**
- point ratios below 0.90 = **50/50**

This is a decisive false positive. The legacy Phase 3A primary comparison is therefore **falsified as insufficiently robust**.

### Improvement A

Add a preregistered nonlinear observable-state feature map to **both** M0 and MH before any history features are added. At minimum this includes quadratic terms for terminal voltage/current/slope/temperature and a small fixed interaction set.

Retest across the same 50 seeds:

- median MH/M0 = **1.0011**
- 5–95% range = **0.9991–1.0030**
- point ratios below 0.90 = **0/50**

The false history advantage disappears.

## Falsification experiment B — blocked acquisition drift

Generator: there is **no true order effect**. All repetitions of each order are acquired as one block. The outcome drifts monotonically with laboratory run index.

Legacy analysis across 50 datasets:

- median MH/M0 = **0.5078**
- 5–95% range = **0.4922–0.5299**
- point ratios below 0.90 = **50/50**

Again the legacy analysis can manufacture a large apparent history benefit.

### Improvement B

The hardware protocol must therefore freeze the acquisition schedule itself, not just A/B/C waveforms. Required changes:

- randomized or balanced interleaving of the six orders within each device;
- recorded `run_index` and `block_id` for every trial;
- run-index terms available to M0 and MH equally;
- an explicit schedule audit that rejects order/run-index confounding;
- no order-by-order acquisition blocks in the confirmatory experiment.

With run-index terms available to both models, the same blocked-drift stress test gives:

- median MH/M0 = **1.0008**
- 5–95% range = **0.9995–1.0024**
- point ratios below 0.90 = **0/50**

## Positive control after hardening

A true order term was then added while retaining nonlinear observable-state effects and randomized acquisition order.

Across 50 datasets under the hardened analysis:

- median MH/M0 = **0.8003**
- 5–95% range = **0.7804–0.8190**
- point ratios below 0.90 = **50/50**

Thus the hardening removes the tested false positives without erasing sensitivity to an actual order contribution.

## Additional design corrections required before hardware lock

The adversarial round also exposed implementation/design issues that must be corrected before `CONFIRMATORY_LOCKED`:

1. **Pre-state failures must be excluded at trial level before model fitting.** The current scaffold only reports gate status; it does not yet remove failed trials from the primary analysis.
2. **Device exclusion must be based on each device's failed-trial fraction**, not on whether a device has any failed trial at all.
3. **Order-specific exclusion imbalance must be audited** to prevent selection-induced artifacts.
4. **Within-gate balance must be checked.** Absolute thresholds alone do not prevent systematic order differences inside the accepted region.
5. **G5 and G6 must be computed, not left `NOT_EVALUATED`.** Memoryless and insensitive-probe controls are required for a final mechanism claim.
6. **G3 must have an explicit strong-baseline path.** A history effect that disappears against a well-specified innovations/state-space or prediction-error model is standard system identification, not an IAT-specific predictive method.
7. **G8 must recompute more than the point RMSE ratio.** The independent validator should reproduce the bootstrap intervals, shuffle result, exclusions, control gates, and hashes.

## Revised next-step sequence

```text
Adversarial falsification r1
    ↓
HARDENED ANALYSIS SPEC
    ↓
1000-run adversarial regression suite
    ↓
8-device RC hardware pilot
    ↓
freeze waveform / reset / probe bands / schedule / gates / transform
    ↓
strong-baseline implementation audit
    ↓
SHA-256 confirmatory lock
    ↓
30 new devices
    ↓
independent validator + falsification controls
```

## Scientific boundary

Even if the future hardware study passes, the strongest justified statement remains operational:

> Earlier stimulus order contributes predictive information about a common-probe terminal response after observable current-state, acquisition-time, conditioning-work, and control adjustments.

That result would not establish a new conserved energy, a quantum mechanism, or superiority over standard dynamic-system identification unless those claims are separately tested.
