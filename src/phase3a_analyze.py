from __future__ import annotations

import argparse
from pathlib import Path

from phase3a_core import (
    analyze_subset, gate_status, load_spec, read_csv, save_json,
    sha256_file, validate_trial_rows, write_csv,
)


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
    primary = [
        r for r in rows
        if r["control_type"] == spec["controls"]["memory_device_label"]
        and r["probe_band"] == spec["controls"]["sensitive_probe_label"]
    ]
    analysis = analyze_subset(primary, spec, args.bootstrap)
    gates = gate_status(primary, analysis, spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "predictions.csv", analysis.pop("predictions"))
    report = {
        "phase": "3A",
        "integrity": integrity,
        "primary": analysis,
        "gates": gates,
        "hashes": {"spec": sha256_file(args.spec), "data": sha256_file(args.data)},
        "boundary": spec["scientific_boundary"],
    }
    save_json(out / "analysis.json", report)
    lines = [
        "# IAT Phase 3A analysis",
        "",
        f"- Devices: {integrity['n_devices']}",
        f"- Trials: {integrity['n_trials']}",
        f"- MH/M0: {analysis['ratio_mh_m0']['estimate']:.6f} "
        f"[{analysis['ratio_mh_m0']['ci_low']:.6f}, {analysis['ratio_mh_m0']['ci_high']:.6f}]",
        f"- Shuffled/ordered: {analysis['shuffle_ordered']['estimate']:.6f} "
        f"[{analysis['shuffle_ordered']['ci_low']:.6f}, {analysis['shuffle_ordered']['ci_high']:.6f}]",
        "",
        "## Gates",
    ] + [f"- {k}: {v}" for k, v in gates.items()] + [
        "", "## Scientific boundary", spec["scientific_boundary"]
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
