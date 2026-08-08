# IAT Phase 3A — Adversarial Falsification r1 → r2

## Status

Synthetic software falsification before the eight-device hardware pilot. This document is **not RC-circuit evidence**.

The point of this stage is not to accumulate supportive simulations. It is to make the analysis fail under plausible alternative explanations, repair the failure, and then attack the repair again.

---

## Round 0 — simple null calibration

The first scaffold passed a 1,000-dataset no-order-effect calibration:

- median `MH/M0 = 1.001015`;
- 0.5–99.5% range `0.998041–1.003447`;
- point ratios below 0.90: `0/1000`.

This showed only that history features do not automatically create a large advantage under an easy null.

---

# Round 1 — falsify the legacy linear baseline

## Falsification A — quadratic observable-state confound

Generator: **no true order effect**. Order is correlated with terminal pre-voltage values that remain inside the provisional ±1 mV gate. The outcome depends quadratically on that already-observable voltage.

Legacy linear M0, 50 independent datasets:

- median `MH/M0 = 0.4050`;
- 5–95% range `0.3911–0.4268`;
- point ratios below 0.90: `50/50`.

**Verdict: the legacy primary comparison is falsified.** Order labels can proxy for omitted nonlinear current-state structure.

### Repair r1

Add quadratic observable-state terms and a small fixed interaction set to both M0 and MH.

Retest:

- median `MH/M0 = 1.0011`;
- 5–95% range `0.9991–1.0030`;
- point ratios below 0.90: `0/50`.

## Falsification B — blocked acquisition drift

Generator: **no true order effect**. All repetitions of each order are acquired in blocks and the outcome drifts with laboratory run index.

Legacy analysis:

- median `MH/M0 = 0.5078`;
- 5–95% range `0.4922–0.5299`;
- point ratios below 0.90: `50/50`.

**Verdict: order-block acquisition is invalid for confirmation.**

### Repair r1-b

The acquisition schedule becomes part of the preregistered intervention protocol:

- balanced randomized interleaving of all six orders within each device;
- no order-by-order acquisition blocks;
- record `run_index` and `block_id`;
- make run-index terms available equally to M0 and MH;
- reject schedules with excessive order/run-index correlation.

Retest with run-index adjustment:

- median `MH/M0 = 1.0008`;
- 5–95% range `0.9995–1.0024`;
- point ratios below 0.90: `0/50`.

At this point r1 looked acceptable. We then attacked r1 itself.

---

# Round 2 — falsify the quadratic hardening

The key question became:

> Is a fixed quadratic M0 genuinely robust, or does it merely survive the exact nonlinearity used to design it?

The answer was the latter.

## Falsification C — smooth oscillatory observable response

Generator: **no true order effect**. The accepted pre-voltage remains observable and order-correlated, but the probe response is a smooth sinusoidal function of that voltage rather than a quadratic.

Quadratic r1 baseline:

- median `MH/M0 = 0.3504`;
- 5–95% range approximately `0.3326–0.3637`;
- point ratios below 0.90: `100%`.

This is an even larger false history advantage than in Round 1.

**Verdict: r1 is falsified as a general current-state baseline.**

## Falsification D — threshold observable response

Generator: **no true order effect**. The outcome changes sharply when the magnitude of accepted pre-voltage crosses a threshold.

Quadratic r1 diagnostic across 30 datasets:

- median `MH/M0 = 0.9002`;
- 5–95% range approximately `0.8768–0.9233`;
- point ratios below 0.90: `43.3%`.

The artifact is weaker than the sine case but still unacceptable because a substantial fraction of datasets cross the nominal 10% support threshold without any true history mechanism.

---

# Repair r2 — physically scaled hinge-spline M0

Instead of guessing a small polynomial family, r2 uses a frozen piecewise-linear hinge-spline basis for the four observable-equivalence coordinates:

- terminal pre-voltage;
- pre-current as fraction of instrument full scale;
- pre-voltage slope;
- pre-temperature deviation.

The spline locations are not fitted from outcomes. They are fixed relative to the preregistered physical gate widths. The same feature map is given to M0 and MH; MH receives history only after that map.

The r2 map also retains:

- acquisition run index and its square;
- block ID;
- conditioning work;
- total stimulus amount;
- last stimulus;
- a small fixed set of physically interpretable interactions.

## Retest C — sine confound under r2

Across 50 datasets:

- median `MH/M0 = 1.0012`;
- 5–95% range `0.9987–1.0054`;
- point ratios below 0.90: `0/50`.

## Retest D — threshold confound under r2

Across 50 datasets:

- median `MH/M0 = 1.0057`;
- 5–95% range `0.9968–1.0200`;
- point ratios below 0.90: `0/50`.

The two Round-2 false advantages disappear.

---

# Positive control after r2

A genuine order term was added while retaining nonlinear observable-state effects and randomized acquisition order.

Across 50 datasets under r2:

- median `MH/M0 = 0.8015`;
- 5–95% range `0.7399–0.8692`;
- point ratio below 0.90 in `96%` of datasets.

Thus the r2 repair removes the tested false positives while retaining useful sensitivity to an actual order contribution.

The positive control is intentionally not treated as a power guarantee for hardware. It only checks that the defense has not made the detector completely inert.

---

# What changed in the formal Phase 3A design

The current `PILOT_OPEN` design now requires:

1. **Spline-hardened M0** rather than the original linear or quadratic-only current-state model.
2. **Trial-level pre-state exclusion before fitting.**
3. **Device exclusion from failed-trial fraction**, not from any single failed trial.
4. **Order-specific exclusion-imbalance audit.**
5. **Within-gate observable-balance audit.**
6. **Balanced randomized acquisition schedule** with `run_index` and `block_id` recorded.
7. **No confirmatory order blocks.**
8. **Memoryless and insensitive-probe Gate paths.**
9. **Numeric energy-residual validation.**
10. **Strong standard system-identification baselines before confirmatory lock.**
11. **Independent validator expansion** to exclusions, intervals, shuffle/control Gates and final Gate table.
12. **Confirmatory refit-bootstrap or a preregistered justified alternative.**

---

# Remaining strongest alternative explanations

Round 2 does not end falsification. The next software attacks should target different failure classes rather than more variations of a one-dimensional pre-voltage function:

- reset carryover from the previous trial;
- sparse/degenerate conditional-shuffle strata;
- device-specific nonlinearities and heterogeneity;
- missing samples, ADC clipping and outliers;
- time-synchronization error between voltage and current;
- temperature drift with long correlation time;
- order-dependent exclusion caused by the pre-state gate;
- inadequacy of the spline basis under multivariate nonlinear response surfaces;
- uncertainty under model refitting rather than fixed-prediction bootstrap.

If one of these creates a false confirmatory PASS, the analysis must be revised again before hardware locking.

---

# Revised execution sequence

```text
simple null calibration
    ↓
adversarial falsification r1
    ↓
quadratic/run-index repair
    ↓
adversarial falsification r2
    ↓
spline/current-state repair
    ↓
software stress suite r3
    ↓
8-device RC pilot
    ↓
freeze waveform / reset / probe bands / schedule / gates / transform
    ↓
strong-baseline audit
    ↓
SHA-256 CONFIRMATORY_LOCKED
    ↓
30 new devices
    ↓
independent validation + falsification controls
```

## Scientific boundary

Even if future hardware passes every Gate, the strongest justified statement remains operational:

> Earlier stimulus order contributes predictive information about a common-probe terminal response after observable current-state, acquisition-time, conditioning-work, and control adjustments.

If standard DelayARX/state-space/prediction-error models achieve the same result, the correct interpretation is that the history value is representable by standard dynamic-system identification.

No result in this software phase establishes a new conserved energy, a quantum mechanism, or a new law of nature.
