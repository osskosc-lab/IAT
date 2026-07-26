#!/usr/bin/env python3
"""Independent validator for IAT Phase 2C-r2A outputs.

This module intentionally does not import r2a_experiment.py.
It reconstructs all Gate estimates from saved CSV/JSON artifacts.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "r2a"
CONFIG_PATH = ROOT / "config" / "r2a_config.yaml"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_lags(text: str) -> tuple[int, ...]:
    text = text.strip()
    if not text:
        return tuple()
    return tuple(int(x) for x in text.split(",") if x)


def geometric_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0.0):
        raise ValueError("Geometric mean requires positive values")
    return float(np.exp(np.mean(np.log(arr))))


def seed_cluster_values(
    rows: list[dict],
    value_fn: Callable[[dict], float],
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["master_seed"]), []).append(float(value_fn(row)))
    return grouped


def bootstrap_seed_stat(
    seed_to_values: dict[int, list[float]],
    aggregate_within_seed: Callable[[list[float]], float],
    aggregate_across_seeds: Callable[[list[float]], float],
    reps: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    seeds = sorted(seed_to_values)
    per_seed = np.asarray(
        [aggregate_within_seed(seed_to_values[s]) for s in seeds],
        dtype=float,
    )
    estimate = float(aggregate_across_seeds(per_seed.tolist()))
    boot = np.empty(reps, dtype=float)
    n = len(per_seed)
    for i in range(reps):
        sample = per_seed[rng.integers(0, n, size=n)]
        boot[i] = float(aggregate_across_seeds(sample.tolist()))
    alpha = 1.0 - ci_level
    low = float(np.quantile(boot, alpha / 2.0))
    high = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return estimate, low, high


def ratio_gate(
    rows: list[dict],
    field: str,
    reps: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    grouped = seed_cluster_values(rows, lambda r: float(r[field]))
    return bootstrap_seed_stat(
        grouped,
        aggregate_within_seed=geometric_mean,
        aggregate_across_seeds=geometric_mean,
        reps=reps,
        ci_level=ci_level,
        rng=rng,
    )


def proportion_gate(
    rows: list[dict],
    value_fn: Callable[[dict], float],
    reps: int,
    ci_level: float,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    grouped = seed_cluster_values(rows, value_fn)
    return bootstrap_seed_stat(
        grouped,
        aggregate_within_seed=lambda x: float(np.mean(x)),
        aggregate_across_seeds=lambda x: float(np.mean(x)),
        reps=reps,
        ci_level=ci_level,
        rng=rng,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = load_config()
    metrics = read_csv(RESULTS / "metrics.csv")
    selections = read_csv(RESULTS / "selection.csv")
    hashes = read_csv(RESULTS / "trajectory_hashes.csv")
    with (RESULTS / "truth_key.json").open("r", encoding="utf-8") as f:
        truth = json.load(f)

    truth_map = {k: tuple(int(x) for x in v) for k, v in truth["scenario_truth"].items()}
    selection_index = {
        (int(r["master_seed"]), r["scenario_id"]): r for r in selections
    }

    candidate_max = int(cfg["model"]["candidate_max_lag"])
    in_support_ids = {
        sid for sid, lags in truth_map.items()
        if lags and max(lags) <= candidate_max
    }
    out_support_ids = {
        sid for sid, lags in truth_map.items()
        if lags and max(lags) > candidate_max
    }
    null_ids = {sid for sid, lags in truth_map.items() if not lags}

    enriched = []
    for m in metrics:
        key = (int(m["master_seed"]), m["scenario_id"])
        s = selection_index[key]
        true_lags = truth_map[m["scenario_id"]]
        selected = parse_lags(s["selected_lags"])
        finite_selected = parse_lags(s["finite_selected_lags"])
        enriched.append(
            {
                **m,
                "true_lags": ",".join(map(str, true_lags)),
                "selected_lags": ",".join(map(str, selected)),
                "finite_selected_lags": ",".join(map(str, finite_selected)),
                "exact_recovery": float(selected == true_lags),
                "finite_exact_recovery": float(finite_selected == true_lags),
                "null_false_positive": float(
                    (len(selected) > 0) or (int(m["misspec_detected"]) == 1)
                ),
            }
        )

    in_rows = [r for r in enriched if r["scenario_id"] in in_support_ids]
    out_rows = [r for r in enriched if r["scenario_id"] in out_support_ids]
    null_rows = [r for r in enriched if r["scenario_id"] in null_ids]

    reps = int(cfg["statistics"]["bootstrap_reps"])
    ci_level = float(cfg["statistics"]["ci_level"])
    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))

    recovery = proportion_gate(
        in_rows, lambda r: r["exact_recovery"], reps, ci_level, rng
    )
    adaptive_delay = ratio_gate(
        in_rows, "adaptive_delay_ratio", reps, ci_level, rng
    )
    out_detection = proportion_gate(
        out_rows, lambda r: float(r["misspec_detected"]), reps, ci_level, rng
    )
    null_fp = proportion_gate(
        null_rows, lambda r: r["null_false_positive"], reps, ci_level, rng
    )
    shuffle_ratio = ratio_gate(
        in_rows, "shuffle_ordered_ratio", reps, ci_level, rng
    )
    finite_recovery = proportion_gate(
        in_rows, lambda r: r["finite_exact_recovery"], reps, ci_level, rng
    )
    finite_ratio = ratio_gate(
        in_rows, "finite_adaptive_delay_ratio", reps, ci_level, rng
    )

    all_hashes = [r["sha256"] for r in hashes]
    expected_metric_rows = len(truth_map) * len({int(r["master_seed"]) for r in metrics})
    gate0_pass = (
        len(metrics) == expected_metric_rows
        and len(selections) == expected_metric_rows
        and len(all_hashes) == len(set(all_hashes))
        and all(float(r["spectral_radius"]) < 1.0 for r in metrics)
        and all(math.isfinite(float(r["adaptive_rmse"])) for r in metrics)
    )

    thresholds = cfg["gates"]
    gate_rows = [
        {
            "gate": "Gate 0",
            "name": "Implementation integrity",
            "estimate": 1.0 if gate0_pass else 0.0,
            "ci_low": "",
            "ci_high": "",
            "criterion": "all integrity checks PASS",
            "pass": gate0_pass,
        },
        {
            "gate": "Gate 1",
            "name": "In-support exact recovery",
            "estimate": recovery[0],
            "ci_low": recovery[1],
            "ci_high": recovery[2],
            "criterion": f"99% CI lower > {thresholds['recovery_ci_lower']}",
            "pass": recovery[1] > float(thresholds["recovery_ci_lower"]),
        },
        {
            "gate": "Gate 2",
            "name": "Adaptive / DelayARX non-inferiority",
            "estimate": adaptive_delay[0],
            "ci_low": adaptive_delay[1],
            "ci_high": adaptive_delay[2],
            "criterion": f"99% CI upper < {thresholds['delay_noninferiority_upper']}",
            "pass": adaptive_delay[2] < float(thresholds["delay_noninferiority_upper"]),
        },
        {
            "gate": "Gate 3",
            "name": "Out-of-support detection sensitivity",
            "estimate": out_detection[0],
            "ci_low": out_detection[1],
            "ci_high": out_detection[2],
            "criterion": f"99% CI lower > {thresholds['out_support_detection_ci_lower']}",
            "pass": out_detection[1] > float(thresholds["out_support_detection_ci_lower"]),
        },
        {
            "gate": "Gate 4",
            "name": "Null false-positive rate",
            "estimate": null_fp[0],
            "ci_low": null_fp[1],
            "ci_high": null_fp[2],
            "criterion": f"99% CI upper < {thresholds['null_false_positive_ci_upper']}",
            "pass": null_fp[2] < float(thresholds["null_false_positive_ci_upper"]),
        },
        {
            "gate": "Gate 5",
            "name": "Temporal-order mechanism",
            "estimate": shuffle_ratio[0],
            "ci_low": shuffle_ratio[1],
            "ci_high": shuffle_ratio[2],
            "criterion": f"99% CI lower > {thresholds['shuffle_ratio_ci_lower']}",
            "pass": shuffle_ratio[1] > float(thresholds["shuffle_ratio_ci_lower"]),
        },
        {
            "gate": "Gate 6a",
            "name": "Finite-sample exact recovery",
            "estimate": finite_recovery[0],
            "ci_low": finite_recovery[1],
            "ci_high": finite_recovery[2],
            "criterion": f"99% CI lower > {thresholds['finite_recovery_ci_lower']}",
            "pass": finite_recovery[1] > float(thresholds["finite_recovery_ci_lower"]),
        },
        {
            "gate": "Gate 6b",
            "name": "Finite-sample / DelayARX non-inferiority",
            "estimate": finite_ratio[0],
            "ci_low": finite_ratio[1],
            "ci_high": finite_ratio[2],
            "criterion": f"99% CI upper < {thresholds['delay_noninferiority_upper']}",
            "pass": finite_ratio[2] < float(thresholds["delay_noninferiority_upper"]),
        },
    ]

    all_pass = all(bool(r["pass"]) for r in gate_rows)
    report = {
        "experiment_id": cfg["experiment_id"],
        "status": "PASS" if all_pass else "PARTIAL_SUPPORT_OR_FAIL",
        "all_gates_pass": all_pass,
        "seed_count": len({int(r["master_seed"]) for r in metrics}),
        "scenario_count": len(truth_map),
        "bootstrap_reps": reps,
        "ci_level": ci_level,
        "in_support_scenarios": sorted(in_support_ids),
        "out_support_scenarios": sorted(out_support_ids),
        "null_scenarios": sorted(null_ids),
        "gates": gate_rows,
        "limitations": [
            "Stable linear-Gaussian delay toy models only.",
            "No real-data, nonlinear, quantum, or new-physical-law claim.",
            "Gate thresholds are Phase 2C-r2A preregistered operational criteria.",
        ],
    }

    with (RESULTS / "validation_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    write_csv(RESULTS / "gate_table.csv", gate_rows)
    write_csv(RESULTS / "validated_rows.csv", enriched)

    labels = [r["gate"] + "\n" + r["name"] for r in gate_rows[1:]]
    estimates = [float(r["estimate"]) for r in gate_rows[1:]]
    lows = [float(r["ci_low"]) for r in gate_rows[1:]]
    highs = [float(r["ci_high"]) for r in gate_rows[1:]]

    fig = plt.figure(figsize=(10, 6))
    y = np.arange(len(labels))
    xerr = np.vstack([
        np.asarray(estimates) - np.asarray(lows),
        np.asarray(highs) - np.asarray(estimates),
    ])
    plt.errorbar(estimates, y, xerr=xerr, fmt="o", capsize=4)
    plt.yticks(y, labels)
    plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Estimate with 99% seed-cluster bootstrap CI")
    plt.title("IAT Phase 2C-r2A confirmatory Gate estimates")
    plt.tight_layout()
    fig.savefig(RESULTS / "gate_estimates.png", dpi=180)
    plt.close(fig)

    summary_lines = [
        "# IAT Phase 2C-r2A Confirmatory Result",
        "",
        f"**Final status: {report['status']}**",
        "",
        f"- Seeds: {report['seed_count']}",
        f"- Scenarios: {report['scenario_count']}",
        f"- Bootstrap: {reps} repetitions, {int(ci_level * 100)}% CI",
        "",
        "## Gate table",
        "",
        "| Gate | Test | Estimate | 99% CI | Criterion | Result |",
        "|---|---|---:|---:|---|---|",
    ]
    for r in gate_rows:
        if r["ci_low"] == "":
            ci_text = "—"
        else:
            ci_text = f"[{float(r['ci_low']):.6f}, {float(r['ci_high']):.6f}]"
        summary_lines.append(
            f"| {r['gate']} | {r['name']} | {float(r['estimate']):.6f} | "
            f"{ci_text} | {r['criterion']} | {'PASS' if r['pass'] else 'FAIL'} |"
        )
    summary_lines += [
        "",
        "## Scientific boundary",
        "",
        "This result concerns only the frozen stable linear-Gaussian delay toy-model family. "
        "It does not establish superiority over correctly specified standard delay models, "
        "generalization to real data or nonlinear systems, a quantum connection, or a new physical law.",
        "",
    ]
    (RESULTS / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    with PdfPages(RESULTS / "IAT_Phase2C_r2A_report.pdf") as pdf:
        fig1 = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        title = "IAT Phase 2C-r2A\nBlind structure recovery and misspecification detection"
        plt.text(0.05, 0.92, title, fontsize=20, weight="bold", va="top")
        plt.text(0.05, 0.80, f"Final status: {report['status']}", fontsize=16, weight="bold")
        plt.text(
            0.05,
            0.73,
            f"{report['seed_count']} confirmatory seeds | "
            f"{reps} bootstrap repetitions | {int(ci_level * 100)}% CI",
            fontsize=11,
        )
        y0 = 0.66
        for i, r in enumerate(gate_rows):
            if r["ci_low"] == "":
                metric = f"{float(r['estimate']):.4f}"
            else:
                metric = (
                    f"{float(r['estimate']):.4f} "
                    f"[{float(r['ci_low']):.4f}, {float(r['ci_high']):.4f}]"
                )
            line = f"{r['gate']}: {'PASS' if r['pass'] else 'FAIL'} — {r['name']} — {metric}"
            plt.text(0.07, y0 - i * 0.055, line, fontsize=10, va="top")
        plt.text(
            0.05,
            0.14,
            "Boundary: stable linear-Gaussian delay toy models only. "
            "No real-data, nonlinear, quantum, or new-law claim.",
            fontsize=10,
            wrap=True,
        )
        pdf.savefig(fig1, bbox_inches="tight")
        plt.close(fig1)

        image = plt.imread(RESULTS / "gate_estimates.png")
        fig2 = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        plt.imshow(image)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not all_pass:
        raise SystemExit("One or more preregistered Gates failed.")


if __name__ == "__main__":
    main()
