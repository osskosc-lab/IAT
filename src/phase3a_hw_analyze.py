"""Frozen Phase 3A-HW v1.0 analysis.

Observable-only analysis of the preregistered eight-device pilot. All prediction
metrics are computed from leave-one-device-out held-out predictions. Feature scaling
is fitted on training devices only and then applied unchanged to the held-out device.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

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
    if not raw:
        raise ValueError("empty dataset")
    missing = set(spec["required_trial_columns"]) - set(raw[0])
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
    best = 0
    current = 0
    previous = None
    for order in orders:
        current = current + 1 if order == previous else 1
        best = max(best, current)
        previous = order
    return best


def integrity_audit(rows: list[dict], schedule_path: str | Path, spec: dict) -> dict:
    expected = build_schedule(schedule_path)
    actual = {r["trial_id"]: r for r in rows}
    reasons: list[str] = []
    if len(rows) != int(spec["hardware_pilot"]["total_trials"]):
        reasons.append(f"trial_count={len(rows)}")
    if len(actual) != len(rows):
        reasons.append("duplicate_trial_id")
    keys = ["device_id", "trial_id", "trial_class", "block_id", "run_index", "order", "previous_trial_order"]
    for e in expected:
        a = actual.get(e["trial_id"])
        if a is None:
            reasons.append(f"missing:{e['trial_id']}")
            continue
        for k in keys:
            if str(a[k]) != str(e[k]):
                reasons.append(f"schedule_mismatch:{e['trial_id']}:{k}")
                break
    devices = sorted({r["device_id"] for r in rows})
    if len(devices) != 8:
        reasons.append(f"devices={len(devices)}")
    for d in devices:
        main = sorted([r for r in rows if r["device_id"] == d and r["trial_class"] == "main"], key=lambda r: r["run_index"])
        longw = [r for r in rows if r["device_id"] == d and r["trial_class"] == "long_washout"]
        if len(main) != 120 or len(longw) != 24:
            reasons.append(f"device_counts:{d}:{len(main)}:{len(longw)}")
        if {o: sum(r["order"] == o for r in main) for o in ("AB", "BA")} != {"AB": 60, "BA": 60}:
            reasons.append(f"order_balance:{d}")
        if _max_run([r["order"] for r in main]) > 3:
            reasons.append(f"run_length:{d}")
        for b in range(1, 7):
            br = [r for r in main if r["block_id"] == b]
            counts = {o: sum(r["order"] == o for r in br) for o in ("AB", "BA")}
            if len(br) != 20 or counts != {"AB": 10, "BA": 10}:
                reasons.append(f"block_balance:{d}:{b}")
    return {
        "pass": not reasons,
        "reasons": reasons[:50],
        "n_trials": len(rows),
        "n_devices": len(devices),
        "generated_schedule_sha256": schedule_sha256(expected),
    }


def _fit_scaler(train: list[dict]) -> dict:
    x = np.array([[float(r[k]) for k in CONT] for r in train], dtype=float)
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    reset = np.array([float(r["reset_elapsed_s"]) for r in train], dtype=float)
    reset_mu = float(reset.mean())
    reset_sd = max(float(reset.std()), 1e-12)
    return {"mu": mu, "sd": sd, "reset_mu": reset_mu, "reset_sd": reset_sd}


def _matrix(rows: list[dict], scaler: dict, *, include_order: bool, include_prev: bool, superflex: bool) -> np.ndarray:
    x = np.array([[float(r[k]) for k in CONT] for r in rows], dtype=float)
    z = (x - scaler["mu"]) / scaler["sd"]
    cols: list[np.ndarray] = [np.ones(len(rows))]
    for j in range(z.shape[1]):
        cols.extend([z[:, j], z[:, j] ** 2])
    for j in [0, 2, 3, 4]:
        for knot in np.arange(-0.8, 0.81, 0.2):
            cols.append(np.maximum(0.0, z[:, j] - knot))
    for a, b in [(0, 2), (0, 4), (2, 4), (0, 3), (2, 3), (3, 4)]:
        cols.append(z[:, a] * z[:, b])
    run = np.array([float(r["run_index"]) for r in rows]) / 144.0
    cols.extend([run, run ** 2])
    for block in range(1, 13):
        cols.append(np.array([float(r["block_id"] == block) for r in rows]))
    if include_prev:
        reset = np.array([float(r["reset_elapsed_s"]) for r in rows])
        rz = (reset - scaler["reset_mu"]) / scaler["reset_sd"]
        cols.extend([rz, rz ** 2])
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
    penalty = np.eye(X.shape[1]) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.pinv(X.T @ X + penalty) @ X.T @ y


def _shuffle_orders(rows: list[dict], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = [dict(r) for r in rows]
    by_block: dict[int, list[int]] = {}
    for i, r in enumerate(out):
        by_block.setdefault(int(r["block_id"]), []).append(i)
    for indices in by_block.values():
        values = np.array([out[i]["order"] for i in indices], dtype=object)
        values = values[rng.permutation(len(values))]
        for i, value in zip(indices, values):
            out[i]["order"] = str(value)
    return out


def cv_pair(rows: list[dict], spec: dict, *, include_prev: bool = True, superflex: bool = False, shuffle_history: bool = False) -> list[dict]:
    alpha = float(spec["models"]["ridge_alpha"])
    devices = sorted({r["device_id"] for r in rows})
    predictions: list[dict] = []
    for fold, test_device in enumerate(devices):
        train = [r for r in rows if r["device_id"] != test_device]
        test = [r for r in rows if r["device_id"] == test_device]
        scaler = _fit_scaler(train)
        y_train = np.array([float(r["probe_work_j"]) for r in train])
        y_test = np.array([float(r["probe_work_j"]) for r in test])
        X0_train = _matrix(train, scaler, include_order=False, include_prev=include_prev, superflex=superflex)
        X0_test = _matrix(test, scaler, include_order=False, include_prev=include_prev, superflex=superflex)
        pred0 = X0_test @ _ridge(X0_train, y_train, alpha)
        Xh_train = _matrix(train, scaler, include_order=True, include_prev=include_prev, superflex=superflex)
        history_test = _shuffle_orders(test, 9200 + fold) if shuffle_history else test
        Xh_test = _matrix(history_test, scaler, include_order=True, include_prev=include_prev, superflex=superflex)
        predh = Xh_test @ _ridge(Xh_train, y_train, alpha)
        for r, y, p0, ph in zip(test, y_test, pred0, predh):
            predictions.append({
                "device_id": r["device_id"], "trial_id": r["trial_id"], "fold": str(fold),
                "sqerr_m0": float((y-p0)**2), "sqerr_mh": float((y-ph)**2),
            })
    return predictions


def cv_previous(rows: list[dict], spec: dict) -> list[dict]:
    alpha = float(spec["models"]["ridge_alpha"])
    devices = sorted({r["device_id"] for r in rows})
    predictions: list[dict] = []
    for fold, test_device in enumerate(devices):
        train = [r for r in rows if r["device_id"] != test_device]
        test = [r for r in rows if r["device_id"] == test_device]
        scaler = _fit_scaler(train)
        y_train = np.array([float(r["probe_work_j"]) for r in train])
        y_test = np.array([float(r["probe_work_j"]) for r in test])
        Xa = _matrix(train, scaler, include_order=False, include_prev=False, superflex=False)
        Xat = _matrix(test, scaler, include_order=False, include_prev=False, superflex=False)
        Xb = _matrix(train, scaler, include_order=False, include_prev=True, superflex=False)
        Xbt = _matrix(test, scaler, include_order=False, include_prev=True, superflex=False)
        pa = Xat @ _ridge(Xa, y_train, alpha)
        pb = Xbt @ _ridge(Xb, y_train, alpha)
        for r, y, a, b in zip(test, y_test, pa, pb):
            predictions.append({
                "device_id": r["device_id"], "trial_id": r["trial_id"], "fold": str(fold),
                "sqerr_base": float((y-a)**2), "sqerr_prev": float((y-b)**2),
            })
    return predictions


def ratio_stats(rows: list[dict], numerator: str, denominator: str, spec: dict) -> dict:
    devices = sorted({r["device_id"] for r in rows})
    sn = np.array([sum(float(r[numerator]) for r in rows if r["device_id"] == d) for d in devices])
    sd = np.array([sum(float(r[denominator]) for r in rows if r["device_id"] == d) for d in devices])
    estimate = math.sqrt(float(sn.sum()/sd.sum()))
    n_boot = int(spec["inference"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(spec["inference"]["bootstrap_seed"]))
    counts = rng.multinomial(len(devices), [1/len(devices)]*len(devices), size=n_boot)
    values = np.sqrt((counts@sn)/np.maximum(counts@sd, 1e-300))
    q = (1-float(spec["inference"]["confidence_level"]))/2
    return {"estimate": estimate, "ci_low": float(np.quantile(values,q)), "ci_high": float(np.quantile(values,1-q))}


def equivalence(rows: list[dict], spec: dict) -> tuple[dict, list[dict]]:
    devices = sorted({r["device_id"] for r in rows})
    n_boot = int(spec["inference"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(spec["inference"]["bootstrap_seed"])+17)
    counts = rng.multinomial(len(devices), [1/len(devices)]*len(devices), size=n_boot)
    q = (1-float(spec["inference"]["confidence_level"]))/2
    margin = float(spec["current_state_equivalence"]["standardized_margin"])
    report: dict[str,dict] = {}
    device_rows: list[dict] = []
    for var in EQ_VARS:
        effects = []
        for d in devices:
            a = np.array([float(r[var]) for r in rows if r["device_id"] == d and r["order"] == "AB"])
            b = np.array([float(r[var]) for r in rows if r["device_id"] == d and r["order"] == "BA"])
            pooled = math.sqrt(max(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2), 1e-24))
            effect = float((a.mean()-b.mean())/pooled)
            effects.append(effect)
            device_rows.append({"device_id":d,"variable":var,"standardized_difference":effect})
        arr = np.array(effects)
        values = (counts@arr)/len(devices)
        lo, hi = float(np.quantile(values,q)), float(np.quantile(values,1-q))
        report[var] = {"estimate":float(arr.mean()),"ci_low":lo,"ci_high":hi,"pass":lo>=-margin and hi<=margin}
    return report, device_rows


def device_consistency(rows: list[dict]) -> dict:
    devices = sorted({r["device_id"] for r in rows})
    ratios = []
    for d in devices:
        rr = [r for r in rows if r["device_id"] == d]
        ratios.append(math.sqrt(sum(float(r["sqerr_mh"]) for r in rr)/max(sum(float(r["sqerr_m0"]) for r in rr),1e-300)))
    return {"device_ratios":dict(zip(devices,ratios)),"median":float(np.median(ratios)),"n_below_1":int(sum(x<1 for x in ratios))}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def classify(gates: dict[str,bool]) -> str:
    priority=[("G0","IMPLEMENTATION_FAILURE"),("G1","CURRENT_STATE_CONFOUND"),("G2","PRIMARY_NULL"),("G3","DEVICE_SPECIFIC"),("G4","ORDER_MECHANISM_FALSIFIED"),("G5","CARRYOVER_ARTIFACT"),("G6","HARDWARE_NULL_FAILURE"),("G7","OBSERVABLE_MODEL_EXPLANATION")]
    for gate,label in priority:
        if not gates[gate]: return label
    return "PILOT_SUPPORT"


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--spec",default="config/phase3a_hw_preregistration.yaml")
    p.add_argument("--schedule",default="config/phase3a_hw_schedule.json")
    p.add_argument("--out",default="results/phase3a_hw")
    args=p.parse_args()
    spec=load_spec(args.spec)
    rows=read_trials(args.data,spec)
    integrity=integrity_audit(rows,args.schedule,spec)
    main_rows=[r for r in rows if r["trial_class"]=="main"]
    long_rows=[r for r in rows if r["trial_class"]=="long_washout"]
    eq,eq_rows=equivalence(main_rows,spec)
    primary=cv_pair(main_rows,spec)
    shuffled=cv_pair(main_rows,spec,shuffle_history=True)
    for a,b in zip(primary,shuffled):
        if a["trial_id"]!=b["trial_id"]: raise RuntimeError("prediction alignment failure")
        a["sqerr_mh_shuffled"]=b["sqerr_mh"]
    previous=cv_previous(main_rows,spec)
    long_predictions=cv_pair(long_rows,spec)
    superflex=cv_pair(main_rows,spec,superflex=True)
    r_primary=ratio_stats(primary,"sqerr_mh","sqerr_m0",spec)
    r_shuffle=ratio_stats(primary,"sqerr_mh_shuffled","sqerr_mh",spec)
    r_previous=ratio_stats(previous,"sqerr_prev","sqerr_base",spec)
    r_long=ratio_stats(long_predictions,"sqerr_mh","sqerr_m0",spec)
    r_super=ratio_stats(superflex,"sqerr_mh","sqerr_m0",spec)
    consistency=device_consistency(primary)
    em=float(spec["inference"]["equivalence_margin"])
    gates={
        "G0":bool(integrity["pass"]),
        "G1":all(v["pass"] for v in eq.values()),
        "G2":r_primary["ci_high"]<float(spec["inference"]["primary_support_threshold"]),
        "G3":consistency["median"]<0.90 and consistency["n_below_1"]>=7,
        "G4":r_shuffle["ci_low"]>float(spec["inference"]["order_shuffle_threshold"]),
        "G5":r_previous["ci_low"]>=1-em and r_previous["ci_high"]<=1+em,
        "G6":r_long["ci_low"]>=1-em and r_long["ci_high"]<=1+em,
        "G7":r_super["ci_high"]<float(spec["inference"]["primary_support_threshold"]),
    }
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"predictions_primary.csv",primary)
    write_csv(out/"predictions_previous.csv",previous)
    write_csv(out/"predictions_long_washout.csv",long_predictions)
    write_csv(out/"predictions_superflex.csv",superflex)
    write_csv(out/"equivalence_device_effects.csv",eq_rows)
    report={
        "phase":spec["phase"],"implementation":"phase3a-hw-v1.0",
        "integrity":integrity,"equivalence":eq,"primary_ratio":r_primary,"shuffle_ratio":r_shuffle,
        "previous_ratio":r_previous,"long_washout_ratio":r_long,"superflex_ratio":r_super,
        "device_consistency":consistency,"gates":gates,"classification":classify(gates),
        "hashes":{"data":sha256(args.data),"spec":sha256(args.spec),"schedule_definition":sha256(args.schedule)},
        "scientific_boundary":spec["scientific_boundary"],
    }
    (out/"analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["# IAT Phase 3A-HW v1.0","",f"Classification: **{report['classification']}**",
           f"Primary MH/M0: {r_primary['estimate']:.6f} [{r_primary['ci_low']:.6f}, {r_primary['ci_high']:.6f}]",
           f"Shuffle/ordered: {r_shuffle['estimate']:.6f} [{r_shuffle['ci_low']:.6f}, {r_shuffle['ci_high']:.6f}]",
           f"Previous-only: {r_previous['estimate']:.6f} [{r_previous['ci_low']:.6f}, {r_previous['ci_high']:.6f}]",
           f"Long-washout: {r_long['estimate']:.6f} [{r_long['ci_low']:.6f}, {r_long['ci_high']:.6f}]",
           f"Superflex: {r_super['estimate']:.6f} [{r_super['ci_low']:.6f}, {r_super['ci_high']:.6f}]","","## Gates"]
    lines += [f"- {g}: {'PASS' if ok else 'FAIL'}" for g,ok in gates.items()]
    lines += ["","## Scientific boundary",spec["scientific_boundary"]]
    (out/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[:18]))


if __name__=="__main__":
    main()
