# IAT Phase 3A preregistration

## Observable-Only History-Conditioned Work Experiment

### Status

**PILOT OPEN.** The confirmatory run is not frozen and must not begin until the eight-device hardware pilot has fixed stimulus amplitudes, durations, reset time, probe bands, measurement ranges, observable-equivalence widths, acquisition schedule, outcome transform/scale, and strong-baseline adapters. Pilot devices and pilot outcomes are permanently excluded from the confirmatory dataset.

The original software scaffold passed a simple null calibration but was subsequently falsified by two adversarial synthetic tests: nonlinear observable-state confounding and acquisition-time drift. The preregistration below incorporates the repairs before any hardware confirmation.

## Central question

> When the observable present looks the same, does a different order of the same past stimuli change the response to the same probe?

The primary claim is deliberately narrower than a claim about a new form of energy. The measured quantity is ordinary terminal work during a common probe,

\[
W_P=\int_0^{T_P}V(t)I(t)\,dt,
\]

and the operational IAT contrast is a difference in this work between order histories. The analysis receives no internal capacitor voltages, latent state, hidden memory coordinate, or simulator truth.

## Hardware and interventions

The first target is a passive multi-time-constant RC ladder. Only terminal voltage, terminal current, and circuit temperature are measured. The six orders are `ABC`, `ACB`, `BAC`, `BCA`, `CAB`, and `CBA`, each repeated eight times per device. Primary pairwise contrasts preserve the last stimulus: `ABC↔BAC`, `ACB↔CAB`, and `BCA↔CBA`.

A small common chirp is the probe. The pilot must freeze one history-sensitive band and one history-insensitive negative-control band. A resistor or fast single-time-constant RC circuit is the memoryless control.

## Acquisition schedule is part of the intervention protocol

The confirmatory study may not acquire all repetitions of one order as a block. A pre-generated balanced randomized interleaving schedule is required within every device.

Every trial records:

- `run_index`;
- `block_id`;
- order label;
- replicate number.

The schedule seed is frozen after the pilot and before any confirmatory device is measured. The schedule audit rejects duplicate run indices and requires the preregistered maximum absolute within-device Spearman correlation between order code and run index to remain at or below 0.10.

This requirement was added because a synthetic no-history generator with ordinary run drift produced a false median `MH/M0≈0.508` when orders were acquired in blocks. Adding run-index adjustment removed the artifact (`≈1.001`).

## Observable pre-state gate and balance audit

A trial is eligible only when the probe-preceding observable state passes all preregistered limits:

- absolute terminal voltage below 1 mV;
- absolute terminal current below 0.1% of full scale;
- absolute one-second voltage slope below 0.2 mV/s;
- absolute temperature deviation below 0.2 °C.

Failed trials are removed **before model fitting**. A device is excluded only when its failed-trial fraction exceeds 10%. The overall excluded-device fraction may not exceed 10%.

Order-specific exclusion is also audited: for each primary pair, the difference in trial-exclusion rate may not exceed 5 percentage points.

Absolute eligibility bounds alone are insufficient because accepted values can still differ systematically by order. Therefore, among accepted trials, each primary pair must also pass a mean-balance audit. For pre-voltage, current fraction, voltage slope and temperature deviation, the absolute pairwise mean difference must be no larger than 25% of the corresponding eligibility width.

This gate is an observable-equivalence statement, not a claim that the hidden internal state is identical.

## Outcome and models

The provisional pilot outcome is `asinh(W_P / 0.001 J)`. The pilot may freeze that scale or preregister a replacement before confirmatory data exist.

### M0 — hardened observable-current-state baseline

`M0` receives:

- observable pre-state variables;
- last stimulus;
- total stimulus amount;
- conditioning work;
- `run_index` and `block_id`.

A fixed nonlinear feature map is included in `M0` before confirmatory locking: squares of pre-voltage, pre-current, pre-voltage slope, temperature deviation and run index, plus a small frozen interaction set. These terms are available equally to `M0` and `MH`.

This repair is mandatory because a synthetic no-history generator with an outcome quadratic in accepted pre-voltage produced a false median `MH/M0≈0.405` under the legacy linear baseline. The hardened observable-state map returned the ratio to `≈1.001` while preserving sensitivity to a true synthetic order effect (`≈0.800`).

### MH — history model

`MH` is exactly the hardened `M0` plus the frozen order representation: positions of A/B/C and the six directed pair-order indicators. The representation cannot be changed after pilot freeze.

Evaluation uses device-grouped nested cross-validation. Trials from a device may never appear in both training and test folds. The primary ratio is

\[
R_{pred}=\frac{RMSE(M_H)}{RMSE(M_0)}.
\]

Support requires the upper bound of a 99% device-cluster interval to be below 0.90. A confirmatory refit-bootstrap or a justified preregistered alternative must be implemented before lock.

## Strong baselines and interpretation

Output-only DelayARX, an innovations state-space model, a prediction-error method, and a small NARX or neural state-space model are preregistered strong baselines. They must produce predictions on the same held-out devices and must be operational before `CONFIRMATORY_LOCKED`.

The IAT history model is not required to beat a well-specified standard dynamic model. If performance is approximately equal, the result is recorded as **standard equivalence**. If the apparent history benefit disappears only because the standard model reconstructs the hidden dynamics from outputs, that is evidence for ordinary system identification rather than an IAT-specific prediction algorithm.

## Falsification operations

1. **Conditional history shuffle:** permute order labels while preserving last stimulus, acquisition block/run strata, conditioning work, and temperature strata. The 99% lower bound of shuffled/ordered RMSE must exceed 1.10.
2. **Memoryless control:** the complete 99% interval for `MH/M0` must lie inside `[0.95, 1.05]`.
3. **Insensitive probe:** the complete 99% interval for `MH/M0` must lie inside `[0.95, 1.05]`.
4. **Conditioning-work adjustment:** both models receive conditioning work, blocking the explanation that one order merely injected more energy.
5. **Acquisition-drift audit:** the randomized schedule and recorded run index must prevent order labels from serving as a proxy for laboratory time.
6. **Within-gate balance audit:** accepted pre-state observables may not remain systematically order-separated beyond preregistered limits.
7. **Energy audit:** absolute unaccounted energy must be below 1% of input work and is checked numerically rather than by trusting a stored Boolean alone.
8. **Independent validator:** must reproduce hashes, exclusions, predictions, confidence intervals, shuffle/control gates and the final Gate table.

## Decision language

A full PASS may support only this statement:

> In the tested passive memory system, earlier stimulus order added predictive information about common-probe terminal work after controlling observable pre-state, acquisition time, final stimulus, total stimulus amount, and conditioning work.

If the history model and standard state-space/DelayARX-type baselines are equivalent, the interpretation is limited further:

> The history value is measurable, but it is representable by standard dynamic-system identification.

The experiment does not establish a new conserved energy, a new physical substance, a quantum connection, or a new law of nature.
