"""Core analysis for IAT Phase 3A.

Only observable trial summaries are used. Device identifiers are used solely for
sample splitting and cluster inference; they are never predictor features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import csv
import hashlib
import json
import math

import numpy as np
import yaml

BASE_CONTINUOUS = [
    "pre_voltage_v",
    "pre_current_a",
    "pre_voltage_slope_v_per_s",
    "pre_temperature_delta_c",
    "total_stimulus_abs",
    "conditioning_work_j",
]
REQUIRED_TRIAL_COLUMNS = {
    "device_id", "trial_id", "order", "replicate", "control_type", "probe_band",
    "last_stimulus", "pre_voltage_v", "pre_current_a",
    "pre_current_fraction_full_scale", "pre_voltage_slope_v_per_s",
    "pre_temperature_delta_c", "total_stimulus_abs", "conditioning_work_j",
    "probe_work_j", "outcome_y", "pre_state_pass", "energy_audit_pass",
}


@dataclass(frozen=True)
class PredictionResult:
    rows: list[dict]
    rmse_m0: float
    rmse_mh: float
    ratio: float


def load_spec(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict], fieldnames: Iterable[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError("fieldnames are required for an empty CSV")
    names = list(fieldnames or rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)


def history_features(order: str) -> np.ndarray:
    if sorted(order) != ["A", "B", "C"] or len(order) != 3:
        raise ValueError(f"invalid order: {order!r}")
    pos = {c: order.index(c) + 1 for c in "ABC"}
    return np.array([
        pos["A"], pos["B"], pos["C"],
        float(pos["A"] < pos["B"]), float(pos["B"] < pos["A"]),
        float(pos["A"] < pos["C"]), float(pos["C"] < pos["A"]),
        float(pos["B"] < pos["C"]), float(pos["C"] < pos["B"]),
    ], dtype=float)


def _to_float(row: dict, key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {key} in trial {row.get('trial_id')}") from exc


def validate_trial_rows(rows: list[dict], spec: dict, require_confirmatory_counts: bool = False) -> dict:
    if not rows:
        raise ValueError("no trial rows")
    missing = REQUIRED_TRIAL_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    trial_ids = [r["trial_id"] for r in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise ValueError("trial_id must be globally unique")
    orders = set(spec["stimuli"]["orders"])
    devices: dict[str, list[dict]] = {}
    for r in rows:
        if r["order"] not in orders:
            raise ValueError(f"unexpected order {r['order']}")
        devices.setdefault(r["device_id"], []).append(r)
        for key in BASE_CONTINUOUS + ["probe_work_j", "outcome_y", "pre_current_fraction_full_scale"]:
            if not np.isfinite(_to_float(r, key)):
                raise ValueError(f"non-finite {key}")
    if require_confirmatory_counts:
        n_expected = int(spec["stimuli"]["confirmatory_devices"])
        if len(devices) != n_expected:
            raise ValueError(f"expected {n_expected} devices, found {len(devices)}")
        reps = int(spec["stimuli"]["repetitions_per_order"])
        for dev, rs in devices.items():
            counts = {o: 0 for o in orders}
            for r in rs:
                counts[r["order"]] += 1
            if any(v != reps for v in counts.values()):
                raise ValueError(f"device {dev} does not have {reps} repetitions per order: {counts}")
    return {"n_rows": len(rows), "n_devices": len(devices), "n_trials": len(trial_ids)}


def _base_matrix(rows: list[dict]) -> np.ndarray:
    cont = np.array([[_to_float(r, k) for k in BASE_CONTINUOUS] for r in rows], dtype=float)
    last = np.array([[float(r["last_stimulus"] == x) for x in ["A", "B", "C"]] for r in rows], dtype=float)
    return np.column_stack([cont, last])


def build_matrix(rows: list[dict], history: bool) -> np.ndarray:
    x = _base_matrix(rows)
    if history:
        x = np.column_stack([x, np.vstack([history_features(r["order"]) for r in rows])])
    return x


def _standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    return (train_x - mean) / sd, (test_x - mean) / sd


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    tx, vx = _standardize(train_x, test_x)
    tx = np.column_stack([np.ones(len(tx)), tx])
    vx = np.column_stack([np.ones(len(vx)), vx])
    penalty = np.eye(tx.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.pinv(tx.T @ tx + penalty) @ (tx.T @ train_y)
    return vx @ beta


def group_folds(devices: list[str], n_folds: int, seed: int) -> list[set[str]]:
    unique = np.array(sorted(set(devices)), dtype=object)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    return [set(unique[i::n_folds].tolist()) for i in range(n_folds)]


def _select_alpha(rows: list[dict], x: np.ndarray, y: np.ndarray, alphas: list[float], n_folds: int, seed: int) -> float:
    devs = [r["device_id"] for r in rows]
    folds = group_folds(devs, min(n_folds, len(set(devs))), seed)
    scores = []
    for alpha in alphas:
        errs = []
        for test_devs in folds:
            te = np.array([d in test_devs for d in devs])
            tr = ~te
            if tr.sum() == 0 or te.sum() == 0:
                continue
            pred = _ridge_predict(x[tr], y[tr], x[te], float(alpha))
            errs.extend((y[te] - pred).tolist())
        scores.append((math.sqrt(float(np.mean(np.square(errs)))), float(alpha)))
    return min(scores)[1]


def cross_validated_predictions(rows: list[dict], spec: dict) -> PredictionResult:
    y = np.array([_to_float(r, "outcome_y") for r in rows], dtype=float)
    x0 = build_matrix(rows, history=False)
    xh = build_matrix(rows, history=True)
    devs = [r["device_id"] for r in rows]
    infer = spec["inference"]
    outer = group_folds(devs, min(int(infer["outer_group_folds"]), len(set(devs))), int(infer["random_seed"]))
    pred0 = np.empty(len(rows), dtype=float)
    predh = np.empty(len(rows), dtype=float)
    fold_id = np.full(len(rows), -1, dtype=int)
    alphas = [float(a) for a in infer["ridge_alphas"]]
    for k, test_devs in enumerate(outer):
        te = np.array([d in test_devs for d in devs])
        tr = ~te
        train_rows = [r for r, keep in zip(rows, tr) if keep]
        a0 = _select_alpha(train_rows, x0[tr], y[tr], alphas, int(infer["inner_group_folds"]), int(infer["random_seed"]) + 100 + k)
        ah = _select_alpha(train_rows, xh[tr], y[tr], alphas, int(infer["inner_group_folds"]), int(infer["random_seed"]) + 200 + k)
        pred0[te] = _ridge_predict(x0[tr], y[tr], x0[te], a0)
        predh[te] = _ridge_predict(xh[tr], y[tr], xh[te], ah)
        fold_id[te] = k
    err0 = y - pred0
    errh = y - predh
    rmse0 = math.sqrt(float(np.mean(err0**2)))
    rmseh = math.sqrt(float(np.mean(errh**2)))
    out = []
    for i, r in enumerate(rows):
        out.append({
            "trial_id": r["trial_id"], "device_id": r["device_id"], "fold": int(fold_id[i]),
            "y": float(y[i]), "pred_m0": float(pred0[i]), "pred_mh": float(predh[i]),
            "sqerr_m0": float(err0[i] ** 2), "sqerr_mh": float(errh[i] ** 2),
        })
    return PredictionResult(out, rmse0, rmseh, rmseh / rmse0)


def bootstrap_ratio(pred_rows: list[dict], numerator_key: str, denominator_key: str, reps: int, confidence: float, seed: int) -> tuple[float, float, float]:
    by_dev: dict[str, list[dict]] = {}
    for r in pred_rows:
        by_dev.setdefault(r["device_id"], []).append(r)
    devices = sorted(by_dev)
    def ratio(sample: list[str]) -> float:
        num, den = [], []
        for d in sample:
            num.extend(float(r[numerator_key]) for r in by_dev[d])
            den.extend(float(r[denominator_key]) for r in by_dev[d])
        return math.sqrt(float(np.mean(num))) / math.sqrt(float(np.mean(den)))
    point = ratio(devices)
    rng = np.random.default_rng(seed)
    vals = np.array([ratio(rng.choice(devices, size=len(devices), replace=True).tolist()) for _ in range(reps)])
    alpha = (1.0 - confidence) / 2.0
    return point, float(np.quantile(vals, alpha)), float(np.quantile(vals, 1.0 - alpha))


def conditional_shuffle(rows: list[dict], seed: int) -> list[dict]:
    """Permute order labels within last-stimulus and coarse observable strata."""
    out = [dict(r) for r in rows]
    work = np.array([_to_float(r, "conditioning_work_j") for r in rows])
    temp = np.array([_to_float(r, "pre_temperature_delta_c") for r in rows])
    wq = np.quantile(work, [0.25, 0.5, 0.75])
    tq = np.quantile(temp, [0.5])
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (r["last_stimulus"], int(np.digitize(work[i], wq)), int(np.digitize(temp[i], tq)))
        strata.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    for idx in strata.values():
        if len(idx) < 2:
            continue
        labels = [rows[i]["order"] for i in idx]
        for _ in range(50):
            perm = rng.permutation(labels).tolist()
            if any(a != b for a, b in zip(labels, perm)):
                break
        for i, label in zip(idx, perm):
            out[i]["order"] = label
    return out


def analyze_subset(rows: list[dict], spec: dict, bootstrap_reps: int | None = None) -> dict:
    pred = cross_validated_predictions(rows, spec)
    infer = spec["inference"]
    reps = int(bootstrap_reps or infer["bootstrap_repetitions"])
    ratio = bootstrap_ratio(pred.rows, "sqerr_mh", "sqerr_m0", reps, float(infer["confidence_level"]), int(infer["random_seed"]))
    shuffled_rows = conditional_shuffle(rows, int(infer["random_seed"]) + 77)
    shuf_pred = cross_validated_predictions(shuffled_rows, spec)
    ordered = {r["trial_id"]: r for r in pred.rows}
    joined = [{
        "device_id": r["device_id"],
        "sqerr_shuffled": r["sqerr_mh"],
        "sqerr_ordered": ordered[r["trial_id"]]["sqerr_mh"],
    } for r in shuf_pred.rows]
    shuffle_ratio = bootstrap_ratio(joined, "sqerr_shuffled", "sqerr_ordered", reps, float(infer["confidence_level"]), int(infer["random_seed"]) + 1)
    return {
        "rmse_m0": pred.rmse_m0,
        "rmse_mh": pred.rmse_mh,
        "ratio_mh_m0": {"estimate": ratio[0], "ci_low": ratio[1], "ci_high": ratio[2]},
        "shuffle_ordered": {"estimate": shuffle_ratio[0], "ci_low": shuffle_ratio[1], "ci_high": shuffle_ratio[2]},
        "predictions": pred.rows,
    }


def gate_status(rows: list[dict], analysis: dict, spec: dict) -> dict:
    thresholds = spec["observable_state_gate"]
    n_devices = len(set(r["device_id"] for r in rows))
    excluded = len(set(r["device_id"] for r in rows if str(r["pre_state_pass"]).lower() not in {"1", "true", "pass"}))
    g1 = excluded / max(n_devices, 1) <= float(thresholds["max_device_exclusion_fraction"])
    ratio = analysis["ratio_mh_m0"]
    shuf = analysis["shuffle_ordered"]
    energy_ok = all(str(r["energy_audit_pass"]).lower() in {"1", "true", "pass"} for r in rows)
    return {
        "G0": "PASS",
        "G1": "PASS" if g1 else "FAIL",
        "G2": "PASS" if ratio["ci_high"] < 0.90 else "FAIL",
        "G3": "NOT_EVALUATED",
        "G4": "PASS" if shuf["ci_low"] > 1.10 else "FAIL",
        "G5": "NOT_EVALUATED",
        "G6": "NOT_EVALUATED",
        "G7": "PASS" if energy_ok else "FAIL",
        "G8": "PENDING_VALIDATOR",
    }


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
