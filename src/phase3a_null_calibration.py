"""Synthetic no-order-effect calibration for IAT Phase 3A."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from phase3a_core import history_features, load_spec

ORDERS = ["ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]


def generate_null(spec: dict, seed: int, n_devices: int | None = None) -> list[dict]:
    """Readable row-form generator used by tests and examples."""
    rng = np.random.default_rng(seed)
    n_dev = int(n_devices or spec["stimuli"]["confirmatory_devices"])
    reps = int(spec["stimuli"]["repetitions_per_order"])
    rows = []
    for d in range(n_dev):
        device_effect = rng.normal(0, 0.20)
        gain = rng.normal(1.0, 0.05)
        for order in ORDERS:
            for rep in range(reps):
                pre_v = rng.normal(0, 2e-4)
                pre_i = rng.normal(0, 1e-4)
                slope = rng.normal(0, 4e-5)
                temp = rng.normal(0, 0.05)
                total = 3.0 + rng.normal(0, 0.02)
                cond_work = 0.05 * gain + rng.normal(0, 0.003)
                y = (0.6 * pre_v / 1e-3 + 0.25 * pre_i / 1e-3 + 0.4 * temp
                     + 0.8 * cond_work + device_effect + rng.normal(0, 0.35))
                trial = f"D{d:03d}_{order}_{rep:02d}"
                rows.append({
                    "device_id": f"D{d:03d}", "trial_id": trial, "order": order,
                    "replicate": str(rep), "control_type": "memory", "probe_band": "sensitive",
                    "last_stimulus": order[-1], "pre_voltage_v": str(pre_v),
                    "pre_current_a": str(pre_i), "pre_current_fraction_full_scale": str(abs(pre_i)),
                    "pre_voltage_slope_v_per_s": str(slope), "pre_temperature_delta_c": str(temp),
                    "total_stimulus_abs": str(total), "conditioning_work_j": str(cond_work),
                    "probe_work_j": str(np.sinh(y) * float(spec["outcome"]["scale_j"])),
                    "outcome_y": str(y), "pre_state_pass": "true", "energy_audit_pass": "true",
                })
    return rows


def _ridge(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float) -> np.ndarray:
    mean = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-12] = 1.0
    a = (train_x - mean) / sd
    b = (test_x - mean) / sd
    a = np.column_stack([np.ones(len(a)), a])
    b = np.column_stack([np.ones(len(b)), b])
    gram = a.T @ a
    gram.flat[:: gram.shape[0] + 1] += alpha
    gram[0, 0] -= alpha
    beta = np.linalg.solve(gram, a.T @ train_y)
    return b @ beta


def _fixed_layout(spec: dict, n_devices: int) -> tuple:
    reps = int(spec["stimuli"]["repetitions_per_order"])
    per_device_orders = np.repeat(np.array(ORDERS, dtype=object), reps)
    orders = np.tile(per_device_orders, n_devices)
    devices = np.repeat(np.arange(n_devices), len(per_device_orders))
    last = np.column_stack([(orders == x).astype(float) for x in "ABC"])
    hist = np.vstack([history_features(str(o)) for o in orders])
    rng = np.random.default_rng(int(spec["inference"]["random_seed"]))
    shuffled = np.arange(n_devices)
    rng.shuffle(shuffled)
    n_folds = int(spec["inference"]["outer_group_folds"])
    fold_devs = [set(shuffled[i::n_folds].tolist()) for i in range(n_folds)]
    folds = [np.isin(devices, list(s)) for s in fold_devs]
    return orders, devices, last, hist, folds


def fast_null_ratio(spec: dict, seed: int, layout: tuple, alpha: float = 1e-4) -> float:
    _, devices, last, hist, folds = layout
    rng = np.random.default_rng(seed)
    n_devices = int(devices.max()) + 1
    n = len(devices)
    device_effect = rng.normal(0, 0.20, n_devices)
    gain = rng.normal(1.0, 0.05, n_devices)
    pre_v = rng.normal(0, 2e-4, n)
    pre_i = rng.normal(0, 1e-4, n)
    slope = rng.normal(0, 4e-5, n)
    temp = rng.normal(0, 0.05, n)
    total = 3.0 + rng.normal(0, 0.02, n)
    cond_work = 0.05 * gain[devices] + rng.normal(0, 0.003, n)
    base = np.column_stack([pre_v, pre_i, slope, temp, total, cond_work, last])
    full = np.column_stack([base, hist])
    y = (0.6 * pre_v / 1e-3 + 0.25 * pre_i / 1e-3 + 0.4 * temp
         + 0.8 * cond_work + device_effect[devices] + rng.normal(0, 0.35, n))
    p0 = np.empty(n)
    ph = np.empty(n)
    for te in folds:
        tr = ~te
        p0[te] = _ridge(base[tr], y[tr], base[te], alpha)
        ph[te] = _ridge(full[tr], y[tr], full[te], alpha)
    return float(np.sqrt(np.mean((y - ph) ** 2)) / np.sqrt(np.mean((y - p0) ** 2)))


def run(spec: dict, replicates: int, seed: int) -> dict:
    layout = _fixed_layout(spec, int(spec["stimuli"]["confirmatory_devices"]))
    ratios = np.array([fast_null_ratio(spec, seed + i, layout) for i in range(replicates)])
    fp = float(np.mean(ratios < 0.90))
    return {
        "replicates": replicates,
        "seed_start": seed,
        "ratio_mean": float(ratios.mean()),
        "ratio_median": float(np.median(ratios)),
        "ratio_q005": float(np.quantile(ratios, 0.005)),
        "ratio_q995": float(np.quantile(ratios, 0.995)),
        "false_positive_rate_point_ratio_below_0_90": fp,
        "calibration_pass": bool(fp <= 0.02 and 0.97 <= np.median(ratios) <= 1.03),
        "ratios": ratios.tolist(),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec", default="config/phase3a_spec.yaml")
    p.add_argument("--replicates", type=int, default=1000)
    p.add_argument("--seed", type=int, default=330000)
    p.add_argument("--out", default="results/phase3a/null_calibration")
    args = p.parse_args()
    spec = load_spec(args.spec)
    result = run(spec, args.replicates, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "null_calibration.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / "null_ratios.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["replicate", "mh_m0_ratio"])
        w.writerows(enumerate(result["ratios"]))
    summary = (
        "# Phase 3A null calibration\n\n"
        f"- Replicates: {result['replicates']}\n"
        f"- Median MH/M0: {result['ratio_median']:.6f}\n"
        f"- 0.5%-99.5% range: [{result['ratio_q005']:.6f}, {result['ratio_q995']:.6f}]\n"
        f"- Point-ratio false positive rate (<0.90): {result['false_positive_rate_point_ratio_below_0_90']:.4f}\n"
        f"- Calibration: {'PASS' if result['calibration_pass'] else 'FAIL'}\n"
    )
    (out / "summary.md").write_text(summary, encoding="utf-8")
    with PdfPages(out / "IAT_Phase3A_Null_Calibration_Report.pdf") as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.08, 0.93, "IAT Phase 3A", fontsize=18, weight="bold")
        fig.text(0.08, 0.89, "Observable-only null calibration", fontsize=15)
        fig.text(0.08, 0.82, summary.replace("# Phase 3A null calibration\n\n", ""), fontsize=11, va="top", family="monospace")
        fig.text(0.08, 0.48, "Interpretation", fontsize=13, weight="bold")
        fig.text(0.08, 0.44, "Under a generator with no order effect, the frozen history representation did not\nproduce a material predictive advantage over the observable-current-state model.\nThis calibrates the software pipeline only; it is not hardware evidence.", fontsize=10, va="top")
        plt.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        fig = plt.figure(figsize=(8.27, 6.0))
        plt.hist(result["ratios"], bins=40)
        plt.axvline(0.90, linestyle="--", label="support threshold 0.90")
        plt.axvline(1.00, linestyle=":", label="null equivalence 1.00")
        plt.xlabel("RMSE(MH) / RMSE(M0)")
        plt.ylabel("Synthetic null replicates")
        plt.title("Phase 3A null-calibration ratio distribution")
        plt.legend()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
    print(summary)
    if not result["calibration_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
