from __future__ import annotations

import argparse
from pathlib import Path

from phase3a_core_hardened_r2 import (
    analyze_subset,
    apply_prestate_policy,
    gate_status,
    load_spec,
    observable_balance_audit,
    read_csv,
    save_json,
    schedule_audit,
    sha256_file,
    validate_trial_rows,
    write_csv,
)


def _subset(rows: list[dict], control_type: str | None = None, probe_band: str | None = None) -> list[dict]:
    out = rows
    if control_type is not None:
        out = [r for r in out if r["control_type"] == control_type]
    if probe_band is not None:
        out = [r for r in out if r["probe_band"] == probe_band]
    return out


def _safe_analysis(rows: list[dict], spec: dict, bootstrap: int | None) -> dict | None:
    if len(set(r["device_id"] for r in rows)) < 2 or len(rows) < 12:
        return None
    return analyze_subset(rows, spec, bootstrap)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/phase3a/trials.csv")
    p.add_argument("--spec", default="config/phase3a_spec.yaml")
    p.add_argument("--out", default="results/phase3a")
    p.add_argument("--bootstrap", type=int, default=None)
    p.add_argument("--confirmatory", action="store_true")
    args = p.parse_args()

    spec = load_spec(args.spec)
    rows = read_csv(args.data)
    integrity = validate_trial_rows(rows, spec, require_confirmatory_counts=args.confirmatory)
    accepted, prestate = apply_prestate_policy(rows, spec)
    schedule = schedule_audit(accepted, spec)

    memory = spec["controls"]["memory_device_label"]
    memoryless = spec["controls"]["memoryless_control_label"]
    sensitive = spec["controls"]["sensitive_probe_label"]
    insensitive = spec["controls"]["insensitive_probe_label"]

    primary = _subset(accepted, memory, sensitive)
    if not primary:
        raise ValueError("no accepted primary memory/sensitive trials")
    balance = observable_balance_audit(primary, spec)
    primary_analysis = analyze_subset(primary, spec, args.bootstrap)

    memoryless_rows = _subset(accepted, memoryless, sensitive)
    insensitive_rows = _subset(accepted, memory, insensitive)
    memoryless_analysis = _safe_analysis(memoryless_rows, spec, args.bootstrap)
    insensitive_analysis = _safe_analysis(insensitive_rows, spec, args.bootstrap)

    gates = gate_status(
        primary,
        primary_analysis,
        spec,
        prestate,
        balance,
        schedule,
        memoryless_analysis,
        insensitive_analysis,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    predictions = primary_analysis.pop("predictions")
    write_csv(out / "predictions.csv", predictions)
    report = {
        "phase": "3A",
        "implementation": "hardened-r2-spline",
        "integrity": integrity,
        "prestate_policy": prestate,
        "schedule_audit": schedule,
        "observable_balance_audit": balance,
        "primary": primary_analysis,
        "memoryless_control": memoryless_analysis,
        "insensitive_probe_control": insensitive_analysis,
        "gates": gates,
        "hashes": {"spec": sha256_file(args.spec), "data": sha256_file(args.data)},
        "boundary": spec["scientific_boundary"],
    }
    save_json(out / "analysis.json", report)

    ratio = primary_analysis["ratio_mh_m0"]
    shuf = primary_analysis["shuffle_ordered"]
    lines = [
        "# IAT Phase 3A hardened-r2 analysis",
        "",
        f"- Raw devices: {integrity['n_devices']}",
        f"- Raw trials: {integrity['n_trials']}",
        f"- Accepted trials after pre-state policy: {prestate['accepted_trials']}",
        f"- MH/M0: {ratio['estimate']:.6f} [{ratio['ci_low']:.6f}, {ratio['ci_high']:.6f}]",
        f"- Shuffled/ordered: {shuf['estimate']:.6f} [{shuf['ci_low']:.6f}, {shuf['ci_high']:.6f}]",
        f"- Schedule audit: {'PASS' if schedule['pass'] else 'FAIL'}",
        f"- Observable balance: {'PASS' if balance['pass'] else 'FAIL'}",
        "",
        "## Gates",
    ] + [f"- {k}: {v}" for k, v in gates.items()] + [
        "", "## Scientific boundary", spec["scientific_boundary"]
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
