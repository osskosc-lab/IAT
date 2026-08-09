"""Independent validator for Phase 3A-HW v1.0 artifacts.

This module intentionally does not import phase3a_hw_analyze. It independently
recomputes hashes, prediction-level RMSE ratios, device-bootstrap intervals,
current-state equivalence intervals, cross-device consistency, Gates, and the
final classification from saved artifacts.
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


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ratio_stats(rows: list[dict], num: str, den: str, spec: dict) -> dict:
    devices = sorted({r["device_id"] for r in rows})
    sn = np.array([sum(float(r[num]) for r in rows if r["device_id"] == d) for d in devices])
    sd = np.array([sum(float(r[den]) for r in rows if r["device_id"] == d) for d in devices])
    estimate = math.sqrt(float(sn.sum()/sd.sum()))
    B = int(spec["inference"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(spec["inference"]["bootstrap_seed"]))
    counts = rng.multinomial(len(devices), [1/len(devices)]*len(devices), size=B)
    vals = np.sqrt((counts@sn)/np.maximum(counts@sd, 1e-300))
    q = (1-float(spec["inference"]["confidence_level"]))/2
    return {"estimate": estimate, "ci_low": float(np.quantile(vals,q)), "ci_high": float(np.quantile(vals,1-q))}


def close_dict(a: dict, b: dict, tol: float=1e-12) -> bool:
    return all(abs(float(a[k])-float(b[k])) <= tol for k in ("estimate","ci_low","ci_high"))


def equivalence(rows: list[dict], spec: dict) -> dict:
    devices = sorted({r["device_id"] for r in rows})
    vars_ = sorted({r["variable"] for r in rows})
    B = int(spec["inference"]["bootstrap_repetitions"])
    rng = np.random.default_rng(int(spec["inference"]["bootstrap_seed"])+17)
    counts = rng.multinomial(len(devices), [1/len(devices)]*len(devices), size=B)
    margin = float(spec["current_state_equivalence"]["standardized_margin"])
    q = (1-float(spec["inference"]["confidence_level"]))/2
    out = {}
    for var in vars_:
        ds = np.array([float(next(r["standardized_difference"] for r in rows if r["device_id"]==d and r["variable"]==var)) for d in devices])
        vals = (counts@ds)/len(devices)
        lo,hi = float(np.quantile(vals,q)),float(np.quantile(vals,1-q))
        out[var] = {"estimate":float(ds.mean()),"ci_low":lo,"ci_high":hi,"pass":lo>=-margin and hi<=margin}
    return out


def consistency(rows: list[dict]) -> dict:
    devices=sorted({r["device_id"] for r in rows}); vals=[]
    for d in devices:
        rr=[r for r in rows if r["device_id"]==d]
        vals.append(math.sqrt(sum(float(r["sqerr_mh"]) for r in rr)/max(sum(float(r["sqerr_m0"]) for r in rr),1e-300)))
    return {"device_ratios":dict(zip(devices,vals)),"median":float(np.median(vals)),"n_below_1":int(sum(v<1 for v in vals))}


def classification(g: dict[str,bool]) -> str:
    order=[("G0","IMPLEMENTATION_FAILURE"),("G1","CURRENT_STATE_CONFOUND"),("G2","PRIMARY_NULL"),("G3","DEVICE_SPECIFIC"),("G4","ORDER_MECHANISM_FALSIFIED"),("G5","CARRYOVER_ARTIFACT"),("G6","HARDWARE_NULL_FAILURE"),("G7","OBSERVABLE_MODEL_EXPLANATION")]
    for gate,label in order:
        if not g[gate]: return label
    return "PILOT_SUPPORT"


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--spec",default="config/phase3a_hw_preregistration.yaml")
    p.add_argument("--schedule",default="config/phase3a_hw_schedule.json")
    p.add_argument("--results",default="results/phase3a_hw")
    args=p.parse_args()
    spec=yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    result_dir=Path(args.results)
    report=json.loads((result_dir/"analysis.json").read_text(encoding="utf-8"))
    primary=read_csv(result_dir/"predictions_primary.csv")
    prev=read_csv(result_dir/"predictions_previous.csv")
    longp=read_csv(result_dir/"predictions_long_washout.csv")
    star=read_csv(result_dir/"predictions_superflex.csv")
    eqrows=read_csv(result_dir/"equivalence_device_effects.csv")
    rpri=ratio_stats(primary,"sqerr_mh","sqerr_m0",spec)
    rsh=ratio_stats(primary,"sqerr_mh_shuffled","sqerr_mh",spec)
    rprev=ratio_stats(prev,"sqerr_prev","sqerr_base",spec)
    rlong=ratio_stats(longp,"sqerr_mh","sqerr_m0",spec)
    rstar=ratio_stats(star,"sqerr_mh","sqerr_m0",spec)
    eq=equivalence(eqrows,spec); cons=consistency(primary)
    margin=float(spec["inference"]["equivalence_margin"])
    expected_schedule=build_schedule(args.schedule)
    g0=(len(expected_schedule)==1152 and schedule_sha256(expected_schedule)==json.loads(Path(args.schedule).read_text(encoding="utf-8"))["generated_schedule_csv_sha256"] and bool(report["integrity"]["pass"]))
    gates={
        "G0":g0,
        "G1":all(v["pass"] for v in eq.values()),
        "G2":rpri["ci_high"]<float(spec["inference"]["primary_support_threshold"]),
        "G3":cons["median"]<0.90 and cons["n_below_1"]>=7,
        "G4":rsh["ci_low"]>float(spec["inference"]["order_shuffle_threshold"]),
        "G5":rprev["ci_low"]>=1-margin and rprev["ci_high"]<=1+margin,
        "G6":rlong["ci_low"]>=1-margin and rlong["ci_high"]<=1+margin,
        "G7":rstar["ci_high"]<float(spec["inference"]["primary_support_threshold"]),
    }
    checks={
        "data_hash":report["hashes"]["data"]==sha256(args.data),
        "spec_hash":report["hashes"]["spec"]==sha256(args.spec),
        "schedule_definition_hash":report["hashes"]["schedule_definition"]==sha256(args.schedule),
        "primary_ratio":close_dict(rpri,report["primary_ratio"]),
        "shuffle_ratio":close_dict(rsh,report["shuffle_ratio"]),
        "previous_ratio":close_dict(rprev,report["previous_ratio"]),
        "long_ratio":close_dict(rlong,report["long_washout_ratio"]),
        "superflex_ratio":close_dict(rstar,report["superflex_ratio"]),
        "equivalence":all(abs(eq[k][x]-report["equivalence"][k][x])<1e-12 for k in eq for x in ("estimate","ci_low","ci_high")),
        "gates":gates==report["gates"],
        "classification":classification(gates)==report["classification"],
        "fold_exclusivity":all(len({r["fold"] for r in primary if r["device_id"]==d})==1 for d in {r["device_id"] for r in primary}),
    }
    ok=all(checks.values())
    out={"status":"PASS" if ok else "FAIL","checks":checks,"recomputed_gates":gates,"classification":classification(gates)}
    (result_dir/"validator.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    print(json.dumps(out,indent=2))
    if not ok: raise SystemExit(1)


if __name__=="__main__":
    main()
