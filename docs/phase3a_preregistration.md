# IAT Phase 3A preregistration

## Observable-Only History-Conditioned Work Experiment

### Status

**PILOT OPEN.** The confirmatory run is not frozen and must not begin until the eight-device hardware pilot has fixed stimulus amplitudes, durations, reset time, probe bands, measurement ranges, observable-equivalence widths, and the outcome transform. Pilot devices and pilot outcomes are permanently excluded from the confirmatory dataset.

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

## Observable pre-state gate

A trial is eligible only when the probe-preceding observable state passes all preregistered limits:

- absolute terminal voltage below 1 mV;
- absolute terminal current below 0.1% of full scale;
- absolute one-second voltage slope below 0.2 mV/s;
- absolute temperature deviation below 0.2 °C.

A device with more than 10% ineligible trials is an implementation failure and is excluded by a rule fixed before outcomes are examined. This gate is an observable-equivalence statement, not a claim that the hidden internal state is identical.

## Outcome and models

The primary outcome is `asinh(W_P / 0.001 J)` unless the pilot freezes a different scale before confirmatory data exist.

`M0` receives observable pre-state variables, last stimulus, total stimulus amount, and conditioning work. `MH` adds only the frozen order representation: positions of A/B/C and the six directed pair-order indicators. The representation cannot be changed after pilot freeze.

Evaluation uses device-grouped nested cross-validation. Trials from a device may never appear in both training and test folds. The primary ratio is

\[
R_{pred}=\frac{RMSE(M_H)}{RMSE(M_0)}.
\]

Support requires the upper bound of a 99% device-cluster bootstrap interval to be below 0.90.

## Strong baselines and interpretation

Output-only DelayARX, an innovations state-space model, a prediction-error method, and a small NARX or neural state-space model are preregistered strong baselines. They must produce predictions using the external prediction schema and must be evaluated on the same held-out devices. Equality with these baselines is recorded as **standard equivalence**, not as failure of the order-history claim and not as IAT-specific algorithmic superiority.

## Falsification operations

1. **Conditional history shuffle:** permute order labels within last-stimulus and observable-state strata. The 99% lower bound of shuffled/ordered RMSE must exceed 1.10.
2. **Memoryless control:** the complete 99% interval for `MH/M0` must lie inside `[0.95, 1.05]`.
3. **Insensitive probe:** the complete 99% interval for `MH/M0` must lie inside `[0.95, 1.05]`.
4. **Conditioning-work adjustment:** both models receive conditioning work, blocking the explanation that one order merely injected more energy.
5. **Energy audit:** absolute unaccounted energy must be below 1% of input work.

## Decision language

A PASS supports only this statement:

> In the tested passive memory system, earlier stimulus order added predictive information about common-probe terminal work after controlling the observable pre-state, final stimulus, total stimulus amount, and conditioning work.

It does not establish a new conserved energy, a new physical substance, a quantum connection, or superiority over standard dynamic-system identification.
