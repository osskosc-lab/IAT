# Nakayama PLC Comprehensive Audit — Scope and Falsification Rules

This audit evaluates seven claims separately. It intentionally does not collapse software quality, numerical convergence, classical covariance, new-physics existence, Noether theory, and quantum process theory into one PASS/FAIL label.

## Claims

1. The public simulator is numerically complete in its entirety.
2. Any interpolation and integration algorithm preserves causality.
3. Numerical outputs are strictly Lorentz-equivariant after a boost.
4. Multi-particle interactions and self-force are correctly processed.
5. A new information field exists.
6. ICQ is a Noether conserved charge.
7. The classical simulator generalizes directly to a quantum process matrix.

## Decision policy

- A successful build or finite unit-test suite never proves numerical completeness.
- One explicit noncausal interpolation or superluminal integration counterexample is sufficient to refute the universal word “any.”
- Analytic covariance within tolerance is distinguished from bitwise equality at finite precision.
- Multi-particle superposition is distinguished from radiation reaction and self-force renormalization.
- Absence of an information-field implementation is not a proof of nonexistence; it means the simulator supplies no evidence for that claim.
- A Noether claim requires an explicit action, continuous symmetry, current, and conservation derivation.
- A CPTP Choi demonstrator is not a general process matrix and does not establish indefinite causal order.

## Upstream pin

The workflow audits `sogebu/special-relativity-web` at commit:

`7fc8cb1e0aeb76eb90a475cd0f2ee64aee321033`

Pinning prevents upstream changes from silently altering the result.
