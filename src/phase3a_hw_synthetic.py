"""Synthetic regression data for Phase 3A-HW software validation only.

These datasets are not hardware evidence. They test whether the frozen pipeline returns
PRIMARY_NULL under no current-order effect, retains sensitivity to a clean current-order
positive control, and detects a previous-trial carryover attack.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from phase3a_hw_schedule import build_schedule


def generate(schedule_path: str, scenario: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    schedule = build_schedule(schedule_path)
    rows: list[dict] = []
    by_device: dict[str, list[dict]] = {}
    for s in schedule:
        by_device.setdefault(s["device_id"], []).append(s)
    for _, (device, sched) in enumerate(sorted(by_device.items())):
        dev_shift = rng.normal(0.0, 0.03)
        local: list[dict] = []
        for s in sched:
            run = int(s["run_index"])
            reset = 10.0 + rng.normal(0, 0.05)
            pre_v = dev_shift + rng.normal(0, 0.02)
            pre_i = 0.25 * pre_v + rng.normal(0, 0.015)
            pre_frac = 0.20 * pre_v + rng.normal(0, 0.02)
            slope = -0.10 * pre_v + rng.normal(0, 0.02)
            temp = rng.normal(0, 0.03) + 0.0002 * run
            row = dict(s)
            row.update({
                "reset_elapsed_s": reset,
                "pre_voltage_v": pre_v,
                "pre_current_a": pre_i,
                "pre_current_fraction_full_scale": pre_frac,
                "pre_voltage_slope_v_per_s": slope,
                "pre_temperature_delta_c": temp,
            })
            local.append(row)
        # Software-control construction: exact AB/BA mean balance for the four
        # preregistered equivalence coordinates. This is not a hardware power claim.
        for var in ["pre_voltage_v", "pre_current_fraction_full_scale", "pre_voltage_slope_v_per_s", "pre_temperature_delta_c"]:
            main_rows = [r for r in local if r["trial_class"] == "main"]
            target = float(np.mean([r[var] for r in main_rows]))
            for order in ("AB", "BA"):
                group = [r for r in main_rows if r["order"] == order]
                shift = float(np.mean([r[var] for r in group])) - target
                for r in group:
                    r[var] -= shift
        for r in local:
            z = (1.2 * r["pre_voltage_v"] - 0.7 * r["pre_current_a"]
                 + 0.6 * r["pre_voltage_v"] ** 2 + 0.2 * np.sin(2.0 * r["pre_temperature_delta_c"]))
            current = 0.0
            previous = 0.0
            if scenario == "positive" and r["trial_class"] == "main":
                current = 0.40 if r["order"] == "AB" else -0.40
            if scenario == "carryover" and r["trial_class"] == "main":
                previous = 0.40 if r["previous_trial_order"] == "AB" else (-0.40 if r["previous_trial_order"] == "BA" else 0.0)
            y = 1.0 + z + 0.001 * float(r["run_index"]) + current + previous + rng.normal(0, 0.12)
            r["probe_work_j"] = float(y)
            rows.append(r)
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--schedule", default="config/phase3a_hw_schedule.json")
    p.add_argument("--scenario", choices=["null", "positive", "carryover"], required=True)
    p.add_argument("--seed", type=int, default=20260809)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = generate(args.schedule, args.scenario, args.seed)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "device_id", "trial_id", "trial_class", "block_id", "run_index", "order", "previous_trial_order",
        "reset_elapsed_s", "pre_voltage_v", "pre_current_a", "pre_current_fraction_full_scale",
        "pre_voltage_slope_v_per_s", "pre_temperature_delta_c", "probe_work_j",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"scenario={args.scenario} rows={len(rows)} out={out}")


if __name__ == "__main__":
    main()
