"""Adversarial software falsification for IAT Phase 3A before hardware locking.

This suite targets two alternative explanations that can create an apparent
history advantage without a genuine within-trial order effect:

1. nonlinear dependence on observable pre-state variables while the baseline is
   only linear;
2. trial-order / laboratory drift confounded with stimulus-order blocks.

It compares the legacy linear baseline with a hardened observable-state feature
map that adds preregistered quadratic terms and run-index terms.  The final
positive control contains a true order effect to verify that hardening does not
remove intended sensitivity.

This is a synthetic implementation audit.  It is not RC-circuit evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ORDERS = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]


def history_features(order: str) -> np.ndarray:
    pos = {c: order.index(c) + 1 for c in "ABC"}
    return np.array([
        pos["A"], pos["B"], pos["C"],
        float(pos["A"] < pos["B"]), float(pos["B"] < pos["A"]),
        float(pos["A"] < pos["C"]), float(pos["C"] < pos["A"]),
        float(pos["B"] < pos["C"]), float(pos["C"] < pos["B"]),
    ], dtype=float)


def _folds(n_devices: int, n_folds: int = 5, seed: int = 20260807) -> list[np.ndarray]:
    dev = np.arange(n_devices)
    rng = np.random.default_rng(seed)
    rng.shuffle(dev)
    return [dev[i::n_folds] for i in range(n_folds)]


def _ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = 1e-4) -> np.ndarray:
    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    a = (train_x - mean) / sd
    b = (test_x - mean) / sd
    a = np.column_stack([np.ones(len(a)), a])
    b = np.column_stack([np.ones(len(b)), b])
    penalty = np.eye(a.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(a.T @ a + penalty) @ (a.T @ train_y)
    return b @ beta


def _design(rows: dict[str, np.ndarray], hardened: bool) -> tuple[np.ndarray, np.ndarray]:
    cont = np.column_stack([
        rows["pre_v"], rows["pre_i"], rows["slope"], rows["temp"],
        rows["total"], rows["work"],
    ])
    parts = [cont]
    if hardened:
        # Fixed before hardware confirmation: nonlinear observable-current-state
        # terms and laboratory run-index terms are available to M0 and MH equally.
        parts.extend([
            cont[:, :4] ** 2,
            cont[:, [0, 1, 3]] * cont[:, [4, 4, 5]],
            rows["run_index"][:, None],
            rows["run_index"][:, None] ** 2,
        ])
    last = np.column_stack([(rows["last"] == x).astype(float) for x in "ABC"])
    x0 = np.column_stack(parts + [last])
    hist = np.vstack([history_features(str(o)) for o in rows["order"]])
    return x0, np.column_stack([x0, hist])


def _cv_ratio(rows: dict[str, np.ndarray], hardened: bool) -> float:
    x0, xh = _design(rows, hardened)
    y = rows["y"]
    device = rows["device"]
    p0 = np.empty(len(y))
    ph = np.empty(len(y))
    for test_devices in _folds(int(device.max()) + 1):
        te = np.isin(device, test_devices)
        tr = ~te
        p0[te] = _ridge(x0[tr], y[tr], x0[te])
        ph[te] = _ridge(xh[tr], y[tr], xh[te])
    return float(np.sqrt(np.mean((y - ph) ** 2)) / np.sqrt(np.mean((y - p0) ** 2)))


def _pack(records: list[tuple]) -> dict[str, np.ndarray]:
    arr = list(zip(*records))
    return {
        "device": np.asarray(arr[0], dtype=int),
        "order": np.asarray(arr[1], dtype=object),
        "last": np.asarray(arr[2], dtype=object),
        "pre_v": np.asarray(arr[3], dtype=float),
        "pre_i": np.asarray(arr[4], dtype=float),
        "slope": np.asarray(arr[5], dtype=float),
        "temp": np.asarray(arr[6], dtype=float),
        "total": np.asarray(arr[7], dtype=float),
        "work": np.asarray(arr[8], dtype=float),
        "y": np.asarray(arr[9], dtype=float),
        "run_index": np.asarray(arr[10], dtype=float),
    }


def nonlinear_observable_confound(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
    """No history effect; order is correlated with a within-gate observable state.

    Outcome is quadratic in pre-voltage.  A linear M0 is misspecified, so order
    labels can act as a proxy for the omitted nonlinearity.
    """
    rng = np.random.default_rng(seed)
    means = dict(zip(ORDERS, [-8e-4, -5e-4, -2e-4, 2e-4, 5e-4, 8e-4]))
    rows = []
    for d in range(n_devices):
        device_effect = rng.normal(0, 0.10)
        for order in ORDERS:
            for rep in range(reps):
                pre_v = rng.normal(means[order], 5e-5)
                pre_i = rng.normal(0, 6e-5)
                slope = rng.normal(0, 3e-5)
                temp = rng.normal(0, 0.03)
                total = 3.0 + rng.normal(0, 0.01)
                work = 0.05 + rng.normal(0, 0.002)
                y = 2.2e6 * pre_v**2 + 0.15 * temp + device_effect + rng.normal(0, 0.18)
                run = rep * len(ORDERS) + ORDERS.index(order)
                rows.append((d, order, order[-1], pre_v, pre_i, slope, temp, total, work, y, run))
    return _pack(rows)


def blocked_run_drift(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
    """No history effect; all repetitions of each order are run as one block."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_devices):
        device_effect = rng.normal(0, 0.10)
        sequence = [o for o in ORDERS for _ in range(reps)]
        for run, order in enumerate(sequence):
            pre_v = rng.normal(0, 1e-4)
            pre_i = rng.normal(0, 5e-5)
            slope = rng.normal(0, 2e-5)
            temp = rng.normal(0, 0.03)
            total = 3.0 + rng.normal(0, 0.01)
            work = 0.05 + rng.normal(0, 0.002)
            y = 0.035 * run + device_effect + rng.normal(0, 0.15)
            rows.append((d, order, order[-1], pre_v, pre_i, slope, temp, total, work, y, run))
    return _pack(rows)


def true_order_effect(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
    """Positive control with a genuine order term under the hardened baseline."""
    rng = np.random.default_rng(seed)
    rows = []
    score = {o: (1.0 if o.index("A") < o.index("B") else -1.0) for o in ORDERS}
    for d in range(n_devices):
        device_effect = rng.normal(0, 0.10)
        sequence = []
        for _ in range(reps):
            block = ORDERS.copy()
            rng.shuffle(block)
            sequence.extend(block)
        for run, order in enumerate(sequence):
            pre_v = rng.normal(0, 1.5e-4)
            pre_i = rng.normal(0, 7e-5)
            slope = rng.normal(0, 3e-5)
            temp = rng.normal(0, 0.04)
            total = 3.0 + rng.normal(0, 0.02)
            work = 0.05 + rng.normal(0, 0.003)
            y = (
                0.6 * pre_v / 1e-3 + 0.2 * pre_i / 1e-3 + 0.3 * temp
                + 1.5e6 * pre_v**2 + 0.35 * score[order]
                + device_effect + rng.normal(0, 0.25)
            )
            rows.append((d, order, order[-1], pre_v, pre_i, slope, temp, total, work, y, run))
    return _pack(rows)


def _summary(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "fraction_point_ratio_below_0_90": float(np.mean(values < 0.90)),
    }


def run(replicates: int = 50, seed_start: int = 0) -> dict:
    seeds = range(seed_start, seed_start + replicates)
    nl_legacy = np.array([_cv_ratio(nonlinear_observable_confound(s), False) for s in seeds])
    seeds = range(seed_start, seed_start + replicates)
    nl_hardened = np.array([_cv_ratio(nonlinear_observable_confound(s), True) for s in seeds])
    seeds = range(seed_start, seed_start + replicates)
    drift_legacy = np.array([_cv_ratio(blocked_run_drift(s), False) for s in seeds])
    seeds = range(seed_start, seed_start + replicates)
    drift_hardened = np.array([_cv_ratio(blocked_run_drift(s), True) for s in seeds])
    seeds = range(seed_start, seed_start + replicates)
    positive = np.array([_cv_ratio(true_order_effect(s), True) for s in seeds])
    result = {
        "experiment": "IAT Phase 3A adversarial falsification r1",
        "replicates": replicates,
        "seed_start": seed_start,
        "scientific_boundary": "synthetic software falsification only; not hardware evidence",
        "nonlinear_observable_confound": {
            "legacy": _summary(nl_legacy),
            "hardened": _summary(nl_hardened),
            "legacy_falsified": bool(np.median(nl_legacy) < 0.90),
            "hardened_pass": bool(np.median(nl_hardened) > 0.97),
        },
        "blocked_run_drift": {
            "legacy": _summary(drift_legacy),
            "hardened": _summary(drift_hardened),
            "legacy_falsified": bool(np.median(drift_legacy) < 0.90),
            "hardened_pass": bool(np.median(drift_hardened) > 0.97),
        },
        "positive_control": {
            "hardened": _summary(positive),
            "sensitivity_preserved": bool(np.median(positive) < 0.90),
        },
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=50)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", default="results/phase3a/adversarial_falsification_r1.json")
    args = p.parse_args()
    result = run(args.replicates, args.seed_start)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not (
        result["nonlinear_observable_confound"]["legacy_falsified"]
        and result["nonlinear_observable_confound"]["hardened_pass"]
        and result["blocked_run_drift"]["legacy_falsified"]
        and result["blocked_run_drift"]["hardened_pass"]
        and result["positive_control"]["sensitivity_preserved"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
