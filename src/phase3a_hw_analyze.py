"""Frozen Phase 3A-HW v1.0 analysis.

The analysis is observable-only. It never receives latent capacitor state or simulator truth.
All model comparisons are evaluated on leave-one-device-out predictions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

from phase3a_hw_schedule import build_schedule, schedule_sha256

CONT = [
    "pre_voltage_v",
    "pre_current_a",
    "pre_current_fraction_full_scale",
    "pre_voltage_slope_v_per_s",
    "pre_temperature_delta_c",
]
EQ_VARS = [
    "pre_voltage_v",
    "pre_current_fraction_full_scale",
    "pre_voltage_slope_v_per_s",
    "pre_temperature_delta_c",
]
NUMERIC = set(CONT + ["block_id", "run_index", "reset_elapsed_s", "probe_work_j"])


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_spec(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_trials(path: str | Path, spec: dict) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    required = set(spec["required_trial_columns"])
    missing = required - set(raw[0] if raw else [])
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for r in raw:
        q = dict(r)
        for k in NUMERIC:
            q[k] = float(q[k])
            if not math.isfinite(q[k]):
                raise ValueError(f"non-finite {k} in {q.get('trial_id')}")
        q["block_id"] = int(q["block_id"])
        q["run_index"] = int(q["run_index"])
        rows.append(q)
    return rows


def _max_run(orders: list[str]) -> int:
    best = cur = 0
    prev = None
    for x in orders:
        cur = cur + 1 if x == prev else 1
        best = max(best, cur)
        prev = x
    return best


def integrity_audit(rows: list[dict], schedule_path: str | Path, spec: dict) -> dict:
    expected = build_schedule(schedule_path)
    key = ["device_id", "trial_id", "trial_class", "block_id", "run_index", "order", "previous_trial_order"]
    actual_map = {r["trial_id"]: r for r in rows}
    reasons: list[str] = []
    if len(rows) != int(spec["hardware_pilot"]["total_trials"]):
        reasons.append(f"trial_count={len(rows)}")
    if len(actual_map) != len(rows):
        reasons.append("duplicate_trial_id")
    for e in expected:
        a = actual_map.get(e["trial_id"])
        if a is None:
            reasons.append(f"missing:{e['trial_id']}")
            continue
        for k in key:
            if str(a[k]) != str(e[k]):
                reasons.append(f"schedule_mismatch:{e['trial_id']}:{k}")
                break
    devices = sorted({r["device_id"] for r in rows})
    if len(devices) != 8:
        reasons.append(f"devices={len(devices)}")
    for d in devices:
        main = sorted([r for r in rows if r["device_id"] == d and r["trial_class"] == "main"], key=lambda x: x["run_index"])
        longw = [r for r in rows if r["device_id"] == d and r["trial_class"] == "long_washout"]
        if len(main) != 120 or len(longw) != 24:
            reasons.append(f"device_counts:{d}:{len(main)}:{len(longw)}")
        counts = {o: sum(r["order"] == o for r in main) for o in ("AB", "BA")}
        if counts != {"AB": 60, "BA": 60}:
            reasons.append(f"order_balance:{d}:{counts}")
        if _max_run([r["order"] for r in main]) > 3:
            reasons.append(f"run_length:{d}")
        for b in range(1, 7):
            br = [r for r in main if r["block_id"] == b]
            bc = {o: sum(r["order"] == o for r in br) for o in ("AB", "BA")}
            if len(br) != 20 or bc != {"AB": 10, "BA": 10}:
                reasons.append(f"block_balance:{d}:{b}:{bc}")
    return {
        "pass": not reasons,
        "reasons": reasons[:50],
        "n_trials": len(rows),
        "n_devices": len(devices),
        "generated_schedule_sha256": schedule_sha256(expected),
    }


def _stats(train: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([[float(r[k]) for k in CONT] for r in train], dtype=float)
    mu = a.mean(axis=0)
    sd = a.std(axis=0, ddof=0)
    sd[sd < 1e-12] = 1.0
    return mu, sd


def _matrix(rows: list[dict], mu: np.ndarray, sd: np.ndarray, *, include_order: bool,
            include_prev: bool, superflex: bool) -> np.ndarray:
    z = (np.array([[float(r[k]) for k in CONT] for r in rows], dtype=float) - mu) / sd
    cols: list[np.ndarray] = [np.ones(len(rows))]
    for j in range(z.shape[1]):
        cols.append(z[:, j])
        cols.append(z[:, j] ** 2)
    # r3-style fixed hinge spline for the four equivalence observables.
    eq_indices = [0, 2, 3, 4]
    for j in eq_indices:
        for knot in np.arange(-0.8, 0.81, 0.2):
            cols.append(np.maximum(0.0, z[:, j] - knot))
    # Frozen low-order interactions.
    for a, b in [(0, 2), (0, 4), (2, 4), (0, 3), (2, 3), (3, 4)]:
        cols.append(z[:, a] * z[:, b])
    run = np.array([float(r["run_index"]) for r in rows]) / 144.0
    cols += [run, run ** 2]
    for b in range(1, 13):
        cols.append(np.array([float(r["block_id"] == b) for r in rows]))
    if include_prev:
        reset = np.array([float(r["reset_elapsed_s"]) for r in rows])
        rscale = max(float(np.std(reset)), 1e-12)
        rz = (reset - float(np.mean(reset))) / rscale
        cols += [rz, rz ** 2]
        for level in ("NONE", "AB", "BA"):
            cols.append(np.array([float(r["previous_trial_order"] == level) for r in rows]))
    if superflex:
        for j in range(z.shape[1]):
            cols.append(z[:, j] ** 3)
        for a in range(z.shape[1]):
            for b in range(a + 1, z.shape[1]):
                cols.append(z[:, a] * z[:, b])
    if include_order:
        cols.append(np.array([float(r["order"] == "AB") for r in rows]))
    return np.column_stack(cols)


def _ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    p = X.shape[1]
    penalty = np.eye(p) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(X.T @ X + penalty) @ X.T @ y


def _shuffle_orders(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [dict(r) for r in rows]
    by_block: dict[int, list[int]] = {}
    for i, r in enumerate(out):
        by_block.setdefault(int(r["block_id"]), []).append(i)
    for idx in by_block.values():
        vals = [out[i]["order"] for i in idx]
        vals = list(np.array(vals, dtype=object)[rng.permutation(len(vals))])
        for i, v in zip(idx, vals):
            out[i]["order"] = str(v)
    return out


def cv_pair(rows: list[dict], spec: dict, *, include_prev: bool = True, superflex: bool = False,
            history: bool = True, shuffle_history: bool = False) -> list[dict]:
    alpha = float(spec["models"]["ridge_alpha"])
    devices = sorted({r["device_id"] for r in rows})
    predictions: list[dict] = []
    for fold, test_device in enumerate(devices):
        train = [r for r in rows if r["device_id"] != test_device]
        test = [r for r in rows if r["device_id"] == test_device]
        mu, sd = _stats(train)
        ytr = np.array([float(r["probe_work_j"]) for r in train])
        yte = np.array([float(r["probe_work_j"]) for r in test])
        X0tr = _matrix(train, mu, sd, include_order=False, include_prev=include_prev, superflex=superflex)
        X0te = _matrix(test, mu, sd, include_order=False, include_prev=include_prev, superflex=superflex)
        b0 = _ridge(X0tr, ytr, alpha)
        p0 = X0te @ b0
        if history:
            Xhtr = _matrix(train, mu, sd, include_order=True, include_prev=include_prev, superflex=superflex)
            test_for_history = _shuffle_orders(test, 9200 + fold) if shuffle_history else test
            Xhte = _matrix(test_for_history, mu, sd, include_order=True, include_prev=include_prev, superflex=superflex)
            bh = _ridge(Xhtr, ytr, alpha)
            ph = Xhte @ bh
        else:
            ph = p0.copy()
        for r, yy, a, b in zip(test, yte, p0, ph):
            predictions.append({
                "device_id": r["device_id"],
                "trial_id": r["trial_id"],
                "fold": str(fold),
                "sqerr_m0": float((yy - a) ** 2),
                "sqerr_mh": float((yy - b) ** 2),
            })
    return predictions


def cv_previous(rows: list[dict], spec: dict) -> list[dict]:
    # Diagnostic model: Mbase omits previous-order/reset; Mprev adds both.
    alpha = float(spec["models"]["ridge_alpha"])
    devices = sorted({r["device_id"] for r in rows})
    out: list[dict] = []
    for fold, d in enumerate(devices):
        train = [r for r in rows if r["device_id"] != d]
        test = [r for r in rows if r["device_id"] == d]
        mu, sd = _stats(train)
        ytr = np.array([float(r["probe_work_j"]) for r in train])
        yte = np.array([float(r["probe_work_j"]) for r in test])
        Xa = _matrix(train, mu, sd, include_order=False, include_prev=False, superflex=False)
        Xat = _matrix(test, mu, sd, include_order=False, include_prev=False, superflex=False)
        Xb = _matrix(train, mu, sd, include_order=False, include_prev=True, superflex=False)
        Xbt = _matrix(test, mu, sd, include_order=False, include_prev=True, superflex=False)
        pa = Xat @ _ridge(Xa, ytr, alpha)
        pb = Xbt @ _ridge(Xb, ytr, alpha)
        for r, yy, a, b in zip(test, yte, pa, pb):
            out.append({"device_id": r["device_id"], "trial_id": r["trial_id"], "fold": str(fold),
                        "sqerr_base": float((yy-a)**2), "sqerr_prev": float((yy-b)**2)})
    return out


def ratio_stats(rows: list[dict], num: str, den: str, spec: dict) -> dict:
    devices = sorted({r["device_id"] for r in rows})
    sn = np.array([sum(float(r[num]) for r in rows if r["device_id"] == d) for d in devices])
    sd = np.array([sum(float(r[den]) for r in rows if r["device_id"] == d) for d in devices])
    estimate = math.sqrt(float(sn.sum() / sd.sum()))
    B = int(spec["inference"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(spec["inference"]["bootstrap_seed"]))
    counts = rng.multinomial(len(devices), [1/len(devices)] * len(devices), size=B)
    vals = np.sqrt((counts @ sn) / np.maximum(counts @ sd, 1e-300))
    q = (1.0 - float(spec["inference"]["confidence_level"])) / 2.0
    return {"estimate": estimate, "ci_low": float(np.quantile(vals, q)), "ci_high": float(np.quantile(vals, 1-q))}


def equivalence(rows: list[dict], spec: dict) -> tuple[dict, list[dict]]:
    devices = sorted({r["device_id"] for r in rows})
    effects: list[dict] = []
    result: dict[str, dict] = {}
    B = int(spec["inference"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(spec["inference"]["bootstrap_seed"]) + 17)
    counts = rng.multinomial(len(devices), [1/len(devices)] * len(devices), size=B)
    margin = float(spec["current_state_equivalence"]["standardized_margin"])
    for var in EQ_VARS:
        ds = []
        for d in devices:
            a = np.array([float(r[var]) for r in rows if r["device_id"] == d and r["order"] == "AB"])
            b = np.array([float(r[var]) for r in rows if r["device_id"] == d and r["order"] == "BA"])
            pooled = math.sqrt(max(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2), 1e-24))
            dd = float((a.mean()-b.mean())/pooled)
            ds.append(dd)
            effects.append({"device_id": d, "variable": var, "standardized_difference": dd})
        dsarr = np.array(ds)
        vals = (counts @ dsarr) / len(devices)
        q = (1.0 - float(spec["inference"]["confidence_level"])) / 2.0
        lo, hi = float(np.quantile(vals, q)), float(np.quantile(vals, 1-q))
        result[var] = {"estimate": float(dsarr.mean()), "ci_low": lo, "ci_high": hi, "pass": lo >= -margin and hi <= margin}
    return result, effects


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def device_consistency(pred: list[dict]) -> dict:
    devices = sorted({r["device_id"] for r in pred})
    vals = []
    for d in devices:
        rr = [r for r in pred if r["device_id"] == d]
        vals.append(math.sqrt(sum(float(r["sqerr_mh"]) for r in rr) / max(sum(float(r["sqerr_m0"]) for r in rr), 1e-300)))
    return {"device_ratios": dict(zip(devices, vals)), "median": float(np.median(vals)), "n_below_1": int(sum(v < 1.0 for v in vals))}


def classify(gates: dict[str, bool]) -> str:
    order = [
        ("G0", "IMPLEMENTATION_FAILURE"), ("G1", "CURRENT_STATE_CONFOUND"),
        ("G2", "PRIMARY_NULL"), ("G3", "DEVICE_SPECIFIC"),
        ("G4", "ORDER_MECHANISM_FALSIFIED"), ("G5", "CARRYOVER_ARTIFACT"),
        ("G6", "HARDWARE_NULL_FAILURE"), ("G7", "OBSERVABLE_MODEL_EXPLANATION"),
    ]
    for g, label in order:
        if not gates[g]:
            return label
    return "PILOT_SUPPORT"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--spec", default="config/phase3a_hw_preregistration.yaml")
    p.add_argument("--schedule", default="config/phase3a_hw_schedule.json")
    p.add_argument("--out", default="results/phase3a_hw")
    args = p.parse_args()
    spec = load_spec(args.spec)
    rows = read_trials(args.data, spec)
    integrity = integrity_audit(rows, args.schedule, spec)
    main_rows = [r for r in rows if r["trial_class"] == "main"]
    long_rows = [r for r in rows if r["trial_class"] == "long_washout"]

    eq, eq_rows = equivalence(main_rows, spec)
    primary = cv_pair(main_rows, spec)
    shuffled = cv_pair(main_rows, spec, shuffle_history=True)
    for a, s in zip(primary, shuffled):
        if a["trial_id"] != s["trial_id"]: raise RuntimeError("prediction alignment failure")
        a["sqerr_mh_shuffled"] = s["sqerr_mh"]
    prev = cv_previous(main_rows, spec)
    longp = cv_pair(long_rows, spec)
    star = cv_pair(main_rows, spec, superflex=True)

    r_primary = ratio_stats(primary, "sqerr_mh", "sqerr_m0", spec)
    r_shuffle = ratio_stats(primary, "sqerr_mh_shuffled", "sqerr_mh", spec)
    r_prev = ratio_stats(prev, "sqerr_prev", "sqerr_base", spec)
    r_long = ratio_stats(longp, "sqerr_mh", "sqerr_m0", spec)
    r_star = ratio_stats(star, "sqerr_mh", "sqerr_m0", spec)
    consistency = device_consistency(primary)
    margin = float(spec["inference"]["equivalence_margin"])
    gates = {
        "G0": bool(integrity["pass"]),
        "G1": all(v["pass"] for v in eq.values()),
        "G2": r_primary["ci_high"] < float(spec["inference"]["primary_support_threshold"]),
        "G3": consistency["median"] < 0.90 and consistency["n_below_1"] >= 7,
        "G4": r_shuffle["ci_low"] > float(spec["inference"]["order_shuffle_threshold"]),
        "G5": r_prev["ci_low"] >= 1-margin and r_prev["ci_high"] <= 1+margin,
        "G6": r_long["ci_low"] >= 1-margin and r_long["ci_high"] <= 1+margin,
        "G7": r_star["ci_high"] < float(spec["inference"]["primary_support_threshold"]),
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"predictions_primary.csv", primary)
    write_csv(out/"predictions_previous.csv", prev)
    write_csv(out/"predictions_long_washout.csv", longp)
    write_csv(out/"predictions_superflex.csv", star)
    write_csv(out/"equivalence_device_effects.csv", eq_rows)
    report = {
        "phase": spec["phase"], "implementation": "phase3a-hw-v1.0",
        "integrity": integrity, "equivalence": eq, "primary_ratio": r_primary,
        "shuffle_ratio": r_shuffle, "previous_ratio": r_prev, "long_washout_ratio": r_long,
        "superflex_ratio": r_star, "device_consistency": consistency,
        "gates": gates, "classification": classify(gates),
        "hashes": {"data": sha256(args.data), "spec": sha256(args.spec), "schedule_definition": sha256(args.schedule)},
        "scientific_boundary": spec["scientific_boundary"],
    }
    (out/"analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = ["# IAT Phase 3A-HW v1.0", "", f"Classification: **{report['classification']}**",
               f"Primary MH/M0: {r_primary['estimate']:.6f} [{r_primary['ci_low']:.6f}, {r_primary['ci_high']:.6f}]",
               f"Shuffle/ordered: {r_shuffle['estimate']:.6f} [{r_shuffle['ci_low']:.6f}, {r_shuffle['ci_high']:.6f}]",
               f"Previous-only diagnostic: {r_prev['estimate']:.6f} [{r_prev['ci_low']:.6f}, {r_prev['ci_high']:.6f}]",
               f"Long-washout MH/M0: {r_long['estimate']:.6f} [{r_long['ci_low']:.6f}, {r_long['ci_high']:.6f}]",
               f"Superflex MH*/M0*: {r_star['estimate']:.6f} [{r_star['ci_low']:.6f}, {r_star['ci_high']:.6f}]", "",
               "## Gates"] + [f"- {g}: {'PASS' if ok else 'FAIL'}" for g, ok in gates.items()] + ["", "## Boundary", spec["scientific_boundary"]]
    (out/"summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    print("\n".join(summary[:12]))


if __name__ == "__main__":
    main()
