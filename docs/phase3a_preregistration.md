# IAT Phase 3A preregistration

## Observable-Only History-Conditioned Work Experiment

### Status

**PILOT OPEN.** The confirmatory run is not frozen and must not begin until the eight-device hardware pilot has fixed stimulus amplitudes, durations, reset time, probe bands, measurement ranges, observable-equivalence widths, acquisition schedule, outcome transform/scale, and strong-baseline adapters. Pilot devices and pilot outcomes are permanently excluded from the confirmatory dataset.

The analysis has undergone three adversarial software-falsification rounds. The legacy linear M0 was falsified by nonlinear observable-state confounding and acquisition drift. A quadratic repair was then falsified by smooth oscillatory and threshold observable response functions. The spline-hardened r2 baseline was subsequently falsified by previous-experimental-trial carryover. The current candidate is **hardened-r3**, which adds explicit previous-trial and reset-time nuisance adjustment.

## Central question

> When the observable present looks the same, does a different order of the same past stimuli change the response to the same probe?

The primary claim is deliberately narrower than a claim about a new form of energy. The measured quantity is ordinary terminal work during a common probe,

\[
W_P=\int_0^{T_P}V(t)I(t)\,dt.
\]

The analysis receives no internal capacitor voltages, latent state, hidden memory coordinate, or simulator truth.

## Hardware and interventions

The first target is a passive multi-time-constant RC ladder. Only terminal voltage, terminal current, and circuit temperature are measured. The six orders are `ABC`, `ACB`, `BAC`, `BCA`, `CAB`, and `CBA`, each repeated eight times per device. Primary pairwise contrasts preserve the last stimulus: `ABC↔BAC`, `ACB↔CAB`, and `BCA↔CBA`.

A small common chirp is the probe. The pilot must freeze one history-sensitive band and one history-insensitive negative-control band. A resistor or fast single-time-constant RC circuit is the memoryless control.

## Acquisition schedule and reset are part of the intervention protocol

The confirmatory study may not acquire all repetitions of one order as a block. A pre-generated balanced randomized interleaving schedule is required within every device.

Every trial records:

- `run_index`;
- `block_id`;
- `reset_elapsed_s`;
- order label and replicate number.

`previous_trial_order` is derived from the actual within-device run-index sequence, rather than trusted from an externally supplied nuisance label.

The schedule seed is frozen after the pilot and before any confirmatory device is measured. The schedule audit rejects duplicate run indices and requires the preregistered maximum absolute within-device Spearman correlation between order code and run index to remain at or below 0.10.

The chosen reset duration is frozen from the eight-device pilot using observable terminal relaxation. Previous-experimental-trial carryover is treated as a nuisance alternative explanation, never as evidence for the current A/B/C history.

This rule is required because a synthetic generator with **no current-order effect** but a previous-trial carryover effect produced a false r2 median `MH/M0≈0.701`. Adding previous-trial order and measured reset elapsed time to both M0 and MH returned the ratio to approximately `1.001` while retaining sensitivity to a simultaneous genuine current-order contribution.

## Observable pre-state gate and balance audit

A trial is eligible only when the probe-preceding observable state passes all preregistered limits:

- absolute terminal voltage below 1 mV;
- absolute terminal current below 0.1% of full scale;
- absolute one-second voltage slope below 0.2 mV/s;
- absolute temperature deviation below 0.2 °C.

Failed trials are removed **before model fitting**. A device is excluded only when its failed-trial fraction exceeds 10%. The overall excluded-device fraction may not exceed 10%.

For each primary pair, the difference in trial-exclusion rate may not exceed 5 percentage points. Among accepted trials, the absolute pairwise mean difference for pre-voltage, current fraction, voltage slope and temperature deviation may not exceed 25% of the corresponding eligibility width.

This gate is an observable-equivalence statement, not a claim that the hidden internal state is identical.

## Outcome

The provisional pilot outcome is `asinh(W_P / 0.001 J)`. The pilot may freeze that scale or preregister a replacement before confirmatory data exist.

## M0 — hardened-r3 observable-current-state baseline

M0 receives:

- observable pre-state variables;
- last stimulus;
- total stimulus amount;
- conditioning work;
- `run_index` and `block_id`;
- `previous_trial_order`;
- `reset_elapsed_s`.

For the four observable-equivalence coordinates—pre-voltage, pre-current fraction of full scale, pre-voltage slope, and temperature deviation—M0 uses a fixed piecewise-linear hinge-spline basis. Knot locations are fixed relative to the physical gate widths at normalized values from `-0.8` through `0.8` in increments of `0.1`; they are not selected using outcome data. A small frozen interaction set, squared run index and squared reset elapsed time are also included.

The identical baseline representation is available to MH.

### Why r3 is required

Software falsification established the following sequence:

- legacy linear M0 under a quadratic observable confound: median `MH/M0≈0.405`;
- quadratic r1 under a smooth sine observable confound: median `≈0.350`;
- quadratic r1 under a threshold confound: median near `0.900`, with substantial false threshold crossing;
- spline r2 under previous-trial carryover: median `≈0.701` despite no current-order effect.

The current r3 repairs returned the tested false advantages to approximately 1 while preserving positive-control sensitivity.

These are implementation stress tests, not hardware power estimates.

## MH — history model

MH is exactly hardened-r3 M0 plus the frozen current-trial order representation: positions of A/B/C and the six directed pair-order indicators. The representation cannot be changed after pilot freeze.

Evaluation uses device-grouped nested cross-validation. Trials from a device may never appear in both training and test folds. The primary ratio is

\[
R_{pred}=\frac{RMSE(M_H)}{RMSE(M_0)}.
\]

Support requires the upper bound of a 99% device-cluster interval to be below 0.90. A confirmatory refit-bootstrap or a justified preregistered alternative must be implemented before lock.

## Strong baselines and interpretation

Output-only DelayARX, an innovations state-space model, a prediction-error method, and a small NARX or neural state-space model are preregistered strong baselines. They must produce predictions on the same held-out devices and must be operational before `CONFIRMATORY_LOCKED`.

The IAT history model is not required to beat a well-specified standard dynamic model. Approximate equality is recorded as **standard equivalence**. If a standard model reconstructs the relevant dynamics from outputs, that is ordinary system identification rather than IAT-specific algorithmic superiority.

## Falsification operations

1. **Conditional history shuffle:** preserve last stimulus, acquisition block/run strata, conditioning work, temperature strata and the carryover nuisance structure while breaking the current order correspondence. The 99% lower bound of shuffled/ordered RMSE must exceed 1.10.
2. **Memoryless control:** the complete 99% interval for `MH/M0` must lie inside `[0.95, 1.05]`.
3. **Insensitive probe:** the complete 99% interval for `MH/M0` must lie inside `[0.95, 1.05]`.
4. **Conditioning-work adjustment:** both models receive conditioning work.
5. **Acquisition-drift audit:** order labels may not proxy laboratory time.
6. **Previous-trial carryover adjustment:** previous-trial order and reset elapsed time are nuisance variables shared by M0 and MH.
7. **Within-gate balance audit:** accepted pre-state observables may not remain systematically order-separated beyond preregistered limits.
8. **Energy audit:** absolute unaccounted energy must be below 1% of input work and is checked numerically.
9. **Independent validator:** must reproduce hashes, exclusions, predictions, confidence intervals, shuffle/control gates and the final Gate table.

## Remaining pre-lock falsification targets

Before confirmatory lock, software stress tests should still attack sparse conditional-shuffle strata, device heterogeneity, missing/clipped/outlier samples, voltage-current timing error, long-memory temperature drift, order-dependent gate exclusion, difficult multivariate nonlinear response surfaces, refit-bootstrap uncertainty, and strong-baseline alignment/leakage.

Any false confirmatory PASS requires another repair/retest round.

## Decision language

A full PASS may support only:

> In the tested passive memory system, earlier stimulus order added predictive information about common-probe terminal work after controlling observable pre-state, acquisition time, previous-trial carryover, final stimulus, total stimulus amount, and conditioning work.

If the history model and standard state-space/DelayARX-type baselines are equivalent:

> The history value is measurable, but it is representable by standard dynamic-system identification.

The experiment does not establish a new conserved energy, a new physical substance, a quantum connection, or a new law of nature.
