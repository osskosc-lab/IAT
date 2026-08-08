"""Hardened core analysis for IAT Phase 3A.

This module preserves the observable-only design while addressing two software
falsifications found before hardware locking:

- nonlinear observable pre-state confounding;
- acquisition-time drift confounded with order blocks.

It also applies pre-state failures before model fitting and evaluates observable
balance and schedule integrity.  This remains PILOT_OPEN code.
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
    "run_index",
]
REQUIRED_TRIAL_COLUMNS = {
    "device_id", "trial_id", "order", "replicate", "run_index", "block_id",
    "previous_trial_order", "reset_elapsed_s",
    "control_type", "probe_band", "last_stimulus", "pre_voltage_v",
    "pre_current_a", "pre_current_fraction_full_scale",
    "pre_voltage_slope_v_per_s", "pre_temperature_delta_c",
    "total_stimulus_abs", "conditioning_work_j", "probe_work_j", "outcome_y",
    "pre_state_pass", "energy_audit_pass", "energy_relative_residual",
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


def _truthy(value: object) -> bool:
    return str(value).lower() in {"1", "true", "pass"}


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
        if r["last_stimulus"] != r["order"][-1]:
            raise ValueError(f"last stimulus/order mismatch in {r['trial_id']}")
        if r["previous_trial_order"] not in orders | {"NONE"}:
            raise ValueError(f"unexpected previous_trial_order {r['previous_trial_order']}")
        devices.setdefault(r["device_id"], []).append(r)
        for key in BASE_CONTINUOUS + [
            "probe_work_j", "outcome_y", "pre_current_fraction_full_scale",
            "energy_relative_residual", "reset_elapsed_s",
        ]:
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


def apply_prestate_policy(rows: list[dict], spec: dict) -> tuple[list[dict], dict]:
    gate = spec["observable_state_gate"]
    by_dev: dict[str, list[dict]] = {}
    for r in rows:
        by_dev.setdefault(r["device_id"], []).append(r)
    max_fail = float(gate["max_failed_trial_fraction_per_device"])
    excluded_devices = set()
    fail_fraction = {}
    for dev, rs in by_dev.items():
        frac = float(np.mean([not _truthy(r["pre_state_pass"]) for r in rs]))
        fail_fraction[dev] = frac
        if frac > max_fail:
            excluded_devices.add(dev)
    accepted = [r for r in rows if r["device_id"] not in excluded_devices and _truthy(r["pre_state_pass"])]

    excluded_device_fraction = len(excluded_devices) / max(len(by_dev), 1)
    orders = list(spec["stimuli"]["orders"])
    total_by_order = {o: 0 for o in orders}
    fail_by_order = {o: 0 for o in orders}
    for r in rows:
        total_by_order[r["order"]] += 1
        if not _truthy(r["pre_state_pass"]):
            fail_by_order[r["order"]] += 1
    fail_rate = {o: fail_by_order[o] / max(total_by_order[o], 1) for o in orders}
    pair_diffs = []
    for a, b in spec["stimuli"]["primary_pairs"]:
        pair_diffs.append(abs(fail_rate[a] - fail_rate[b]))
    max_pair_diff = max(pair_diffs) if pair_diffs else 0.0
    audit = {
        "excluded_devices": sorted(excluded_devices),
        "excluded_device_fraction": excluded_device_fraction,
        "failed_trial_fraction_by_device": fail_fraction,
        "failed_trial_rate_by_order": fail_rate,
        "max_primary_pair_exclusion_rate_difference": max_pair_diff,
        "accepted_trials": len(accepted),
        "pass": (
            excluded_device_fraction <= float(gate["max_device_exclusion_fraction"])
            and max_pair_diff <= float(gate["max_order_pair_exclusion_rate_difference"])
            and len(accepted) > 0
        ),
    }
    return accepted, audit


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def schedule_audit(rows: list[dict], spec: dict) -> dict:
    order_code = {o: i for i, o in enumerate(spec["stimuli"]["orders"])}
    corrs = []
    duplicate_runs = []
    by_dev: dict[str, list[dict]] = {}
    for r in rows:
        by_dev.setdefault(r["device_id"], []).append(r)
    for dev, rs in by_dev.items():
        runs = np.array([_to_float(r, "run_index") for r in rs], dtype=float)
        codes = np.array([order_code[r["order"]] for r in rs], dtype=float)
        if len(set(runs.tolist())) != len(runs):
            duplicate_runs.append(dev)
        rr = _rankdata(runs)
        rc = _rankdata(codes)
        if rr.std() > 0 and rc.std() > 0:
            corrs.append(float(np.corrcoef(rr, rc)[0, 1]))
    max_abs = max((abs(x) for x in corrs), default=0.0)
    limit = float(spec["randomization"]["max_abs_spearman_order_code_vs_run_index"])
    return {
        "max_abs_device_spearman_order_vs_run": max_abs,
        "limit": limit,
        "duplicate_run_index_devices": duplicate_runs,
        "pass": max_abs <= limit and not duplicate_runs,
    }


def observable_balance_audit(rows: list[dict], spec: dict) -> dict:
    cfg = spec["observable_balance_gate"]["max_pair_mean_difference_fraction_of_gate"]
    widths = {
        "pre_voltage_v": float(spec["observable_state_gate"]["abs_pre_voltage_v_max"]),
        "pre_current_fraction_full_scale": float(spec["observable_state_gate"]["abs_pre_current_fraction_full_scale_max"]),
        "pre_voltage_slope_v_per_s": float(spec["observable_state_gate"]["abs_pre_voltage_slope_v_per_s_max"]),
        "pre_temperature_delta_c": float(spec["observable_state_gate"]["abs_pre_temperature_delta_c_max"]),
    }
    details = []
    passed = True
    for a, b in spec["stimuli"]["primary_pairs"]:
        ra = [r for r in rows if r["order"] == a]
        rb = [r for r in rows if r["order"] == b]
        if not ra or not rb:
            passed = False
            continue
        for key, fraction in cfg.items():
            diff = abs(np.mean([_to_float(r, key) for r in ra]) - np.mean([_to_float(r, key) for r in rb]))
            limit = float(fraction) * widths[key]
            ok = diff <= limit
            passed = passed and ok
            details.append({"pair": f"{a}:{b}", "variable": key, "abs_mean_difference": float(diff), "limit": float(limit), "pass": ok})
    return {"pass": bool(passed), "details": details}


def _base_matrix(rows: list[dict]) -> np.ndarray:
    cont = np.array([[_to_float(r, k) for k in BASE_CONTINUOUS] for r in rows], dtype=float)
    pre_v, pre_i, slope, temp, total, work, run = [cont[:, i] for i in range(cont.shape[1])]
    nonlinear = np.column_stack([
        pre_v**2, pre_i**2, slope**2, temp**2, run**2,
        pre_v * total, pre_i * total, temp * work,
    ])
    last = np.array([[float(r["last_stimulus"] == x) for x in ["A", "B", "C"]] for r in rows], dtype=float)
    block_levels = sorted(set(r["block_id"] for r in rows))
    block = np.array([[float(r["block_id"] == x) for x in block_levels] for r in rows], dtype=float)
    return np.column_stack([cont, nonlinear, last, block])


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
        if not errs:
            raise ValueError("insufficient devices for inner grouped CV")
        scores.append((math.sqrt(float(np.mean(np.square(errs)))), float(alpha)))
    return min(scores)[1]


def cross_validated_predictions(rows: list[dict], spec: dict) -> PredictionResult:
    if len(set(r["device_id"] for r in rows)) < 2:
        raise ValueError("at least two devices are required")
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
    """Permute order labels while preserving major observable/time strata."""
    out = [dict(r) for r in rows]
    work = np.array([_to_float(r, "conditioning_work_j") for r in rows])
    temp = np.array([_to_float(r, "pre_temperature_delta_c") for r in rows])
    run = np.array([_to_float(r, "run_index") for r in rows])
    wq = np.quantile(work, [0.25, 0.5, 0.75])
    tq = np.quantile(temp, [0.5])
    rq = np.quantile(run, [0.25, 0.5, 0.75])
    strata: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        key = (
            r["last_stimulus"], r["block_id"], int(np.digitize(work[i], wq)),
            int(np.digitize(temp[i], tq)), int(np.digitize(run[i], rq)),
        )
        strata.setdefault(key, []).append(i)
    rng = np.random.default_rng(seed)
    changed = 0
    for idx in strata.values():
        if len(idx) < 2:
            continue
        labels = [rows[i]["order"] for i in idx]
        perm = labels
        for _ in range(100):
            candidate = rng.permutation(labels).tolist()
            if any(a != b for a, b in zip(labels, candidate)):
                perm = candidate
                break
        for i, label, original in zip(idx, perm, labels):
            out[i]["order"] = label
            changed += int(label != original)
    if changed == 0:
        raise ValueError("conditional shuffle changed no order labels; strata are too sparse")
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


def equivalence_pass(analysis: dict | None, low: float = 0.95, high: float = 1.05) -> bool:
    if analysis is None:
        return False
    ratio = analysis["ratio_mh_m0"]
    return ratio["ci_low"] >= low and ratio["ci_high"] <= high


def gate_status(
    primary_rows: list[dict],
    primary_analysis: dict,
    spec: dict,
    prestate_audit: dict,
    balance_audit: dict,
    schedule: dict,
    memoryless_analysis: dict | None = None,
    insensitive_analysis: dict | None = None,
) -> dict:
    ratio = primary_analysis["ratio_mh_m0"]
    shuf = primary_analysis["shuffle_ordered"]
    threshold = float(spec["energy_audit"]["max_relative_residual"])
    energy_ok = all(abs(_to_float(r, "energy_relative_residual")) < threshold for r in primary_rows)
    return {
        "G0": "PASS" if schedule["pass"] else "FAIL",
        "G1": "PASS" if prestate_audit["pass"] and balance_audit["pass"] else "FAIL",
        "G2": "PASS" if ratio["ci_high"] < 0.90 else "FAIL",
        "G3": "NOT_EVALUATED",
        "G4": "PASS" if shuf["ci_low"] > 1.10 else "FAIL",
        "G5": "PASS" if equivalence_pass(memoryless_analysis) else ("NOT_EVALUATED" if memoryless_analysis is None else "FAIL"),
        "G6": "PASS" if equivalence_pass(insensitive_analysis) else ("NOT_EVALUATED" if insensitive_analysis is None else "FAIL"),
        "G7": "PASS" if energy_ok else "FAIL",
        "G8": "PENDING_VALIDATOR",
    }


def save_json(path: str | Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
