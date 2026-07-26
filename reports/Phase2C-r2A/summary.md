# IAT Phase 2C-r2A Confirmatory Result

**Final status: PASS**

- Seeds: 30
- Scenarios: 5
- Bootstrap: 4,000 repetitions, 99% CI
- GitHub Actions run: 30186022296
- Artifact: IAT-Phase2C-r2A-results

## Gate table

| Gate | Test | Estimate | 99% CI | Criterion | Result |
|---|---|---:|---:|---|---|
| Gate 0 | Implementation integrity | 1.000000 | — | all integrity checks PASS | PASS |
| Gate 1 | In-support exact recovery | 1.000000 | [1.000000, 1.000000] | 99% CI lower > 0.90 | PASS |
| Gate 2 | Adaptive / DelayARX non-inferiority | 0.999932 | [0.999905, 0.999954] | 99% CI upper < 1.05 | PASS |
| Gate 3 | Out-of-support detection sensitivity | 1.000000 | [1.000000, 1.000000] | 99% CI lower > 0.90 | PASS |
| Gate 4 | Null false-positive rate | 0.000000 | [0.000000, 0.000000] | 99% CI upper < 0.10 | PASS |
| Gate 5 | Temporal-order mechanism | 2.218653 | [2.207394, 2.228913] | 99% CI lower > 1.10 | PASS |
| Gate 6a | Finite-sample exact recovery | 1.000000 | [1.000000, 1.000000] | 99% CI lower > 0.80 | PASS |
| Gate 6b | Finite-sample / DelayARX non-inferiority | 0.999922 | [0.999904, 0.999941] | 99% CI upper < 1.05 | PASS |

## Interpretation

The frozen adaptive model recovered all in-support one- and two-lag structures, detected the out-of-support lag condition, produced no false positive under the Markov null, and showed a large performance loss when temporal correspondence was destroyed. Its predictive error was effectively identical to the correctly specified DelayARX baseline.

## Scientific boundary

This result concerns only the frozen stable linear-Gaussian delay toy-model family. It does not establish superiority over correctly specified standard delay models, generalization to real data or nonlinear systems, a quantum connection, or a new physical law.
