# IAT Phase 3A — Adversarial Falsification r1 → r3

## Status

Synthetic software falsification before the eight-device hardware pilot. This document is **not RC-circuit evidence**.

The procedure is deliberately cyclic:

```text
falsify → identify alternative explanation → repair → retest → attack the repair again
```

## Round 0 — simple null calibration

The first scaffold passed a 1,000-dataset no-order-effect calibration:

- median `MH/M0 = 1.001015`;
- 0.5–99.5% range `0.998041–1.003447`;
- point ratios below 0.90: `0/1000`.

This established only that frozen history features do not automatically create a large advantage under an easy null.

---

# Round 1 — legacy linear baseline

## A. Quadratic observable-state confound

No true order effect. Accepted terminal pre-voltage is order-correlated and the outcome depends quadratically on that observable voltage.

Legacy linear M0, 50 datasets:

- median `MH/M0 = 0.4050`;
- 5–95% `0.3911–0.4268`;
- below 0.90: `50/50`.

**Legacy M0 falsified.** Order labels can proxy for omitted nonlinear current-state structure.

### Repair r1

Add quadratic observable-state terms and a fixed small interaction set to M0 and MH equally.

Retest:

- median `1.0011`;
- 5–95% `0.9991–1.0030`;
- below 0.90: `0/50`.

## B. Blocked acquisition drift

No true order effect. Orders are acquired in blocks and the outcome drifts with run index.

Legacy M0:

- median `0.5078`;
- 5–95% `0.4922–0.5299`;
- below 0.90: `50/50`.

**Order-block acquisition falsified.**

### Repair r1-b

- balanced randomized interleaving within device;
- no order-by-order confirmatory blocks;
- record `run_index` and `block_id`;
- run-index terms available to M0 and MH;
- explicit schedule audit.

Retest: median `1.0008`, below 0.90 `0/50`.

---

# Round 2 — attack the quadratic repair

## C. Smooth sinusoidal observable response

No true order effect. Outcome is a smooth sinusoidal function of accepted pre-voltage.

Quadratic r1:

- median `0.3504`;
- 5–95% approximately `0.3326–0.3637`;
- below 0.90: `100%`.

**r1 falsified.** A quadratic model solved only the nonlinearity it was designed against.

## D. Threshold observable response

No true order effect. Outcome changes sharply when accepted pre-voltage crosses a threshold.

Quadratic r1 diagnostic:

- median `0.9002`;
- 5–95% approximately `0.8768–0.9233`;
- below 0.90: `43.3%`.

### Repair r2 — physically scaled hinge-spline M0

Use a frozen piecewise-linear hinge-spline basis for terminal pre-voltage, current fraction of full scale, pre-voltage slope and pre-temperature deviation. Knot locations are fixed relative to the physical gate widths and never selected from outcomes. M0 and MH receive the same current-state representation.

Retests on 50 datasets:

- sine confound r2: median `1.0012`, 5–95% `0.9987–1.0054`, below 0.90 `0/50`;
- threshold confound r2: median `1.0057`, 5–95% `0.9968–1.0200`, below 0.90 `0/50`.

Positive control with a genuine current-order contribution:

- median `0.8015`;
- 5–95% `0.7399–0.8692`;
- below 0.90 in `96%` of datasets.

---

# Round 3 — previous experimental-trial carryover

Round 2 handled current-trial observable state, but it did not distinguish the current A/B/C history from residual effects of the **previous experimental trial**.

## E. Reset-carryover confound

Generator: there is **no effect of the current A/B/C order**. The only structured contribution to the probe response comes from the previous experimental trial's order. A balanced but serially structured acquisition schedule links previous and current order strongly enough that the current history code can act as a proxy.

Under r2, 50 independent datasets gave:

- median `MH/M0 = 0.7008`;
- 5–95% `0.6442–0.7209`;
- below 0.90: `50/50`.

**r2 is falsified against previous-trial carryover.** A large apparent current-history benefit can be produced even though the current A/B/C sequence has no effect at all.

### Repair r3 — explicit carryover nuisance adjustment

The acquisition/analysis specification now requires:

- `previous_trial_order`, derived from actual `run_index` rather than trusted from a supplied label;
- measured `reset_elapsed_s` for every trial;
- previous-trial order one-hot terms in M0 and MH;
- reset elapsed time and its square in M0 and MH;
- hardware pilot evidence that the chosen reset duration is compatible with terminal relaxation;
- previous-trial carryover may never be interpreted as current-history evidence.

Retest on the same class of no-current-history generators:

- median `MH/M0 = 1.00135`;
- 5–95% `0.99778–1.00464`;
- below 0.90: `0/50`.

The tested false history advantage disappears.

## Positive control after r3

A genuine current-order effect and a simultaneous previous-trial carryover term were both generated. After adjusting the previous trial explicitly:

- median `MH/M0 = 0.7617`;
- 5–95% `0.6543–0.8806`;
- below 0.90 in `96%` of datasets.

The r3 defense therefore removes the tested carryover artifact without making the detector inert to an actual current-order contribution.

---

# Current Phase 3A design after three falsification rounds

The `PILOT_OPEN` specification now requires:

1. spline-hardened observable-current-state M0;
2. balanced randomized acquisition with `run_index` and `block_id`;
3. previous-trial order derived from acquisition order;
4. measured reset elapsed time and explicit carryover adjustment;
5. trial-level pre-state exclusion before fitting;
6. device exclusion from failed-trial fraction;
7. order-specific exclusion-imbalance audit;
8. within-gate observable-balance audit;
9. memoryless and insensitive-probe control Gates;
10. numeric energy-residual audit;
11. strong standard system-identification baselines before confirmatory lock;
12. independent validation of exclusions, predictions, intervals, controls and hashes;
13. confirmatory refit-bootstrap or a separately justified preregistered alternative.

## Remaining attacks before hardware lock

The next software stress suite should target failure classes not yet repaired:

- sparse or degenerate conditional-shuffle strata;
- device-specific nonlinear heterogeneity;
- missing samples, ADC clipping and outliers;
- voltage/current time-synchronization error in work integration;
- long-memory temperature drift;
- order-dependent pre-state exclusion near Gate boundaries;
- more difficult multivariate nonlinear observable surfaces;
- fixed-prediction versus refit-bootstrap uncertainty;
- strong-baseline prediction alignment and data leakage.

A false confirmatory PASS in any of these requires another repair/retest round.

## Next empirical stage

```text
software falsification r0-r3
    ↓
software stress suite r4
    ↓
8-device RC hardware pilot
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

> Earlier stimulus order contributes predictive information about a common-probe terminal response after observable current-state, acquisition-time, previous-trial carryover, conditioning-work, and control adjustments.

If standard DelayARX/state-space/prediction-error models achieve the same result, the correct interpretation is that the history value is representable by standard dynamic-system identification.

No result in this software phase establishes a new conserved energy, a quantum mechanism, or a new law of nature.
