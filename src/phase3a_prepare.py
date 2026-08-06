"""Convert long-form electrical samples into one observable-only row per trial."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

from phase3a_core import load_spec, write_csv


def aggregate(raw_path: str, spec_path: str) -> list[dict]:
    spec = load_spec(spec_path)
    with open(raw_path, newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    required = set(spec["raw_sample_schema"]["required_columns"])
    if not raw or required - set(raw[0]):
        raise ValueError(f"missing raw columns: {sorted(required - set(raw[0] if raw else []))}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in raw:
        groups[(r["device_id"], r["trial_id"])].append(r)
    device_pre_temp: dict[str, list[float]] = defaultdict(list)
    provisional: list[dict] = []
    scale = float(spec["outcome"]["scale_j"])
    gate = spec["observable_state_gate"]
    for (device, trial), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: float(r["time_s"]))
        pre = [r for r in rows if r["phase"] == "pre"]
        probe = [r for r in rows if r["phase"] == "probe"]
        if len(pre) < 2 or len(probe) < 2:
            raise ValueError(f"trial {trial} needs at least two pre and probe samples")
        pt = np.array([float(r["time_s"]) for r in pre])
        pv = np.array([float(r["voltage_v"]) for r in pre])
        pi = np.array([float(r["current_a"]) for r in pre])
        ptemp = np.array([float(r["temperature_c"]) for r in pre])
        qt = np.array([float(r["time_s"]) for r in probe])
        qv = np.array([float(r["voltage_v"]) for r in probe])
        qi = np.array([float(r["current_a"]) for r in probe])
        slope = float(np.polyfit(pt - pt.mean(), pv, 1)[0])
        work = float(np.trapezoid(qv * qi, qt))
        first = rows[0]
        full_scale = max(abs(float(x.get("current_full_scale_a", 1.0))) for x in rows)
        row = {
            "device_id": device,
            "trial_id": trial,
            "order": first["order"],
            "replicate": first["replicate"],
            "control_type": first["control_type"],
            "probe_band": first["probe_band"],
            "last_stimulus": first["last_stimulus"],
            "pre_voltage_v": float(pv.mean()),
            "pre_current_a": float(pi.mean()),
            "pre_current_fraction_full_scale": abs(float(pi.mean())) / full_scale,
            "pre_voltage_slope_v_per_s": slope,
            "pre_temperature_c": float(ptemp.mean()),
            "pre_temperature_delta_c": 0.0,
            "total_stimulus_abs": float(first["total_stimulus_abs"]),
            "conditioning_work_j": float(first["conditioning_work_j"]),
            "probe_work_j": work,
            "outcome_y": float(np.arcsinh(work / scale)),
            "probe_max_abs_current_a": float(np.max(np.abs(qi))),
            "probe_end_voltage_v": float(qv[-1]),
            "energy_relative_residual": float(first.get("energy_relative_residual", 0.0) or 0.0),
            "pre_state_pass": False,
            "energy_audit_pass": False,
        }
        device_pre_temp[device].append(row["pre_temperature_c"])
        provisional.append(row)
    medians = {d: float(np.median(v)) for d, v in device_pre_temp.items()}
    for r in provisional:
        r["pre_temperature_delta_c"] = r["pre_temperature_c"] - medians[r["device_id"]]
        r["pre_state_pass"] = (
            abs(r["pre_voltage_v"]) < float(gate["abs_pre_voltage_v_max"])
            and abs(r["pre_current_fraction_full_scale"]) < float(gate["abs_pre_current_fraction_full_scale_max"])
            and abs(r["pre_voltage_slope_v_per_s"]) < float(gate["abs_pre_voltage_slope_v_per_s_max"])
            and abs(r["pre_temperature_delta_c"]) < float(gate["abs_pre_temperature_delta_c_max"])
        )
        r["energy_audit_pass"] = abs(r["energy_relative_residual"]) < float(spec["energy_audit"]["max_relative_residual"])
    return provisional


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True)
    p.add_argument("--spec", default="config/phase3a_spec.yaml")
    p.add_argument("--out", default="data/phase3a/trials.csv")
    args = p.parse_args()
    rows = aggregate(args.raw, args.spec)
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} trial summaries to {args.out}")


if __name__ == "__main__":
    main()
