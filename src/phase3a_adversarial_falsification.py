"""Adversarial software falsification for IAT Phase 3A before hardware locking.

The suite intentionally searches for false history advantages under generators
with no genuine within-trial order effect.  It keeps three analysis variants:

- legacy: linear observable-current-state baseline;
- r1: quadratic/interactions + acquisition run terms;
- r2: physically-scaled hinge-spline current-state baseline + run terms.

Round 1 falsified the legacy baseline.  Round 2 then falsified the quadratic r1
baseline using smooth oscillatory and threshold response functions of observable
pre-state variables.  r2 is the current PILOT_OPEN candidate.

Synthetic audits are not RC-circuit evidence.
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


def _ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
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


def _base_legacy(rows: dict[str, np.ndarray]) -> np.ndarray:
    cont = np.column_stack([
        rows["pre_v"], rows["pre_i"], rows["slope"], rows["temp"],
        rows["total"], rows["work"],
    ])
    last = np.column_stack([(rows["last"] == x).astype(float) for x in "ABC"])
    return np.column_stack([cont, last])


def _base_r1(rows: dict[str, np.ndarray]) -> np.ndarray:
    cont = np.column_stack([
        rows["pre_v"], rows["pre_i"], rows["slope"], rows["temp"],
        rows["total"], rows["work"],
    ])
    nonlinear = np.column_stack([
        cont[:, :4] ** 2,
        cont[:, [0, 1, 3]] * cont[:, [4, 4, 5]],
        rows["run_index"][:, None],
        rows["run_index"][:, None] ** 2,
    ])
    last = np.column_stack([(rows["last"] == x).astype(float) for x in "ABC"])
    return np.column_stack([cont, nonlinear, last])


def _hinges(z: np.ndarray, knots: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z[:, None] - knots[None, :])


def _base_r2(rows: dict[str, np.ndarray]) -> np.ndarray:
    # Fixed physical scaling uses the provisional observable-equivalence widths.
    v = rows["pre_v"] / 1.0e-3
    i = rows["pre_i"] / 1.0e-3
    slope = rows["slope"] / 2.0e-4
    temp = rows["temp"] / 2.0e-1
    total = rows["total"]
    work = rows["work"]
    run = rows["run_index"]
    cont = np.column_stack([v, i, slope, temp, total, work, run])
    knots = np.linspace(-0.8, 0.8, 17)
    spline = np.column_stack([
        _hinges(v, knots), _hinges(i, knots), _hinges(slope, knots), _hinges(temp, knots),
    ])
    interactions = np.column_stack([
        v * i, v * temp, i * temp, v * slope, i * slope, slope * temp,
        rows["pre_v"] * total, rows["pre_i"] * total, rows["temp"] * work,
        run**2,
    ])
    last = np.column_stack([(rows["last"] == x).astype(float) for x in "ABC"])
    return np.column_stack([cont, spline, interactions, last])


def _design(rows: dict[str, np.ndarray], revision: str) -> tuple[np.ndarray, np.ndarray, float]:
    if revision == "legacy":
        x0, alpha = _base_legacy(rows), 1e-4
    elif revision == "r1":
        x0, alpha = _base_r1(rows), 1e-4
    elif revision == "r2":
        x0, alpha = _base_r2(rows), 1e-2
    else:
        raise ValueError(f"unknown revision: {revision}")
    hist = np.vstack([history_features(str(o)) for o in rows["order"]])
    return x0, np.column_stack([x0, hist]), alpha


def _cv_ratio(rows: dict[str, np.ndarray], revision: str | bool) -> float:
    # bool compatibility keeps existing tests readable: False=legacy, True=r1.
    if revision is False:
        revision = "legacy"
    elif revision is True:
        revision = "r1"
    x0, xh, alpha = _design(rows, str(revision))
    y = rows["y"]
    device = rows["device"]
    p0 = np.empty(len(y))
    ph = np.empty(len(y))
    for test_devices in _folds(int(device.max()) + 1):
        te = np.isin(device, test_devices)
        tr = ~te
        p0[te] = _ridge(x0[tr], y[tr], x0[te], alpha)
        ph[te] = _ridge(xh[tr], y[tr], xh[te], alpha)
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


def quadratic_observable_confound(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = dict(zip(ORDERS, [-8e-4, -5e-4, -2e-4, 2e-4, 5e-4, 8e-4]))
    rows = []
    for d in range(n_devices):
        device_effect = rng.normal(0, 0.10)
        for order in ORDERS:
            for rep in range(reps):
                v = rng.normal(means[order], 5e-5)
                i = rng.normal(0, 6e-5)
                slope = rng.normal(0, 3e-5)
                temp = rng.normal(0, 0.03)
                total = 3.0 + rng.normal(0, 0.01)
                work = 0.05 + rng.normal(0, 0.002)
                y = 2.2e6 * v**2 + 0.15 * temp + device_effect + rng.normal(0, 0.18)
                run = rep * len(ORDERS) + ORDERS.index(order)
                rows.append((d, order, order[-1], v, i, slope, temp, total, work, y, run))
    return _pack(rows)


# Backward-compatible name used by tests.
nonlinear_observable_confound = quadratic_observable_confound


def sine_observable_confound(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = dict(zip(ORDERS, [-8e-4, -5e-4, -2e-4, 2e-4, 5e-4, 8e-4]))
    rows = []
    for d in range(n_devices):
        device_effect = rng.normal(0, 0.10)
        for order in ORDERS:
            for rep in range(reps):
                v = rng.normal(means[order], 5e-5)
                i = rng.normal(0, 6e-5)
                slope = rng.normal(0, 3e-5)
                temp = rng.normal(0, 0.03)
                total = 3.0 + rng.normal(0, 0.01)
                work = 0.05 + rng.normal(0, 0.002)
                y = np.sin((v / 1e-3) * 5.5) + 0.15 * temp + device_effect + rng.normal(0, 0.18)
                run = rep * len(ORDERS) + ORDERS.index(order)
                rows.append((d, order, order[-1], v, i, slope, temp, total, work, y, run))
    return _pack(rows)


def threshold_observable_confound(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = dict(zip(ORDERS, [-8e-4, -5e-4, -2e-4, 2e-4, 5e-4, 8e-4]))
    rows = []
    for d in range(n_devices):
        device_effect = rng.normal(0, 0.10)
        for order in ORDERS:
            for rep in range(reps):
                v = rng.normal(means[order], 5e-5)
                i = rng.normal(0, 6e-5)
                slope = rng.normal(0, 3e-5)
                temp = rng.normal(0, 0.03)
                total = 3.0 + rng.normal(0, 0.01)
                work = 0.05 + rng.normal(0, 0.002)
                y = (1.2 if abs(v) > 4e-4 else -0.2) + 0.2 * float(v > 0) + device_effect + rng.normal(0, 0.18)
                run = rep * len(ORDERS) + ORDERS.index(order)
                rows.append((d, order, order[-1], v, i, slope, temp, total, work, y, run))
    return _pack(rows)


def blocked_run_drift(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_devices):
        device_effect = rng.normal(0, 0.10)
        sequence = [o for o in ORDERS for _ in range(reps)]
        for run, order in enumerate(sequence):
            v = rng.normal(0, 1e-4)
            i = rng.normal(0, 5e-5)
            slope = rng.normal(0, 2e-5)
            temp = rng.normal(0, 0.03)
            total = 3.0 + rng.normal(0, 0.01)
            work = 0.05 + rng.normal(0, 0.002)
            y = 0.035 * run + device_effect + rng.normal(0, 0.15)
            rows.append((d, order, order[-1], v, i, slope, temp, total, work, y, run))
    return _pack(rows)


def true_order_effect(seed: int, n_devices: int = 30, reps: int = 8) -> dict[str, np.ndarray]:
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
            v = rng.normal(0, 1.5e-4)
            i = rng.normal(0, 7e-5)
            slope = rng.normal(0, 3e-5)
            temp = rng.normal(0, 0.04)
            total = 3.0 + rng.normal(0, 0.02)
            work = 0.05 + rng.normal(0, 0.003)
            y = (
                0.6 * v / 1e-3 + 0.2 * i / 1e-3 + 0.3 * temp
                + 1.5e6 * v**2 + 0.35 * score[order]
                + device_effect + rng.normal(0, 0.25)
            )
            rows.append((d, order, order[-1], v, i, slope, temp, total, work, y, run))
    return _pack(rows)


def _summary(values: np.ndarray) -> dict:
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "fraction_point_ratio_below_0_90": float(np.mean(values < 0.90)),
    }


def _many(generator, revision: str, seeds: range) -> np.ndarray:
    return np.array([_cv_ratio(generator(s), revision) for s in seeds])


def run(replicates: int = 20, seed_start: int = 0) -> dict:
    seed_range = lambda: range(seed_start, seed_start + replicates)
    quad_legacy = _many(quadratic_observable_confound, "legacy", seed_range())
    quad_r1 = _many(quadratic_observable_confound, "r1", seed_range())
    drift_legacy = _many(blocked_run_drift, "legacy", seed_range())
    drift_r1 = _many(blocked_run_drift, "r1", seed_range())
    sine_r1 = _many(sine_observable_confound, "r1", seed_range())
    sine_r2 = _many(sine_observable_confound, "r2", seed_range())
    threshold_r1 = _many(threshold_observable_confound, "r1", seed_range())
    threshold_r2 = _many(threshold_observable_confound, "r2", seed_range())
    positive_r2 = _many(true_order_effect, "r2", seed_range())
    result = {
        "experiment": "IAT Phase 3A adversarial falsification r2",
        "replicates": replicates,
        "seed_start": seed_start,
        "scientific_boundary": "synthetic software falsification only; not hardware evidence",
        "round1_quadratic_observable_confound": {
            "legacy": _summary(quad_legacy),
            "r1": _summary(quad_r1),
        },
        "round1_blocked_run_drift": {
            "legacy": _summary(drift_legacy),
            "r1": _summary(drift_r1),
        },
        "round2_sine_observable_confound": {
            "r1": _summary(sine_r1),
            "r2": _summary(sine_r2),
        },
        "round2_threshold_observable_confound": {
            "r1": _summary(threshold_r1),
            "r2": _summary(threshold_r2),
        },
        "positive_control_r2": _summary(positive_r2),
        "r2_pass": bool(
            np.median(sine_r2) > 0.97
            and np.median(threshold_r2) > 0.97
            and np.mean(sine_r2 < 0.90) == 0
            and np.mean(threshold_r2 < 0.90) == 0
            and np.median(positive_r2) < 0.90
        ),
    }
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replicates", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--out", default="results/phase3a/adversarial_falsification_r2.json")
    args = p.parse_args()
    result = run(args.replicates, args.seed_start)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["r2_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
