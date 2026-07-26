#!/usr/bin/env python3
"""IAT Phase 2C-r2A blind structure-recovery experiment.

This is a model-specific simulation experiment for stable linear-Gaussian
delay systems. It does not test real data, nonlinear systems, quantum maps,
or a new physical law.

Scientific safeguards:
- true lag sets are used only by the generator and final truth-key export;
- train / inner / outer / diagnostic / test trajectories are disjoint;
- test data are never used for alpha or structure selection;
- pilot and confirmatory seeds are deterministic and committed;
- all trajectory arrays are hashed;
- an independent validator reads CSV outputs without importing this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "r2a_config.yaml"
SEED_PATH = ROOT / "config" / "r2a_confirmatory_seeds.json"


@dataclass(frozen=True)
class Scenario:
    public_id: str
    true_lags: dict[int, float]


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_seeds() -> list[int]:
    with SEED_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return [int(x) for x in payload["seeds"]]


def scenario_table(cfg: dict) -> list[Scenario]:
    return [
        Scenario(public_id=s["id"], true_lags={int(k): float(v) for k, v in s["true_lags"].items()})
        for s in cfg["scenarios"]
    ]


def stable_radius(a: float, lag_coeffs: dict[int, float]) -> float:
    max_lag = max(lag_coeffs, default=0)
    if max_lag == 0:
        return abs(a)
    mat = np.zeros((max_lag + 1, max_lag + 1), dtype=float)
    mat[0, 0] = a
    for lag, coef in lag_coeffs.items():
        mat[0, lag] = coef
    mat[1:, :-1] = np.eye(max_lag)
    return float(np.max(np.abs(np.linalg.eigvals(mat))))


def seed_for(master_seed: int, scenario_index: int, split_index: int, trajectory_index: int) -> int:
    raw = f"{master_seed}|{scenario_index}|{split_index}|{trajectory_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**32 - 1)


def make_input(kind: str, amplitude: float, n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=float)
    if kind == "sine":
        freq = rng.uniform(0.004, 0.07)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        return amplitude * np.sin(2.0 * np.pi * freq * t + phase)
    if kind == "prbs":
        block = int(rng.integers(4, 25))
        values = rng.choice(np.array([-amplitude, amplitude]), size=int(math.ceil(n / block)))
        return np.repeat(values, block)[:n].astype(float)
    if kind == "step":
        onset = int(rng.integers(max(8, n // 8), max(9, n // 3)))
        u = np.zeros(n, dtype=float)
        u[onset:] = amplitude
        return u
    if kind == "pulse":
        onset = int(rng.integers(max(8, n // 8), max(9, n // 3)))
        width = int(rng.integers(max(8, n // 16), max(9, n // 5)))
        u = np.zeros(n, dtype=float)
        u[onset:min(n, onset + width)] = amplitude
        return u
    if kind == "chirp":
        f0 = rng.uniform(0.002, 0.01)
        f1 = rng.uniform(0.06, 0.14)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        duration = max(1.0, float(n - 1))
        cycles = f0 * t + 0.5 * (f1 - f0) * (t**2) / duration
        return amplitude * np.sin(2.0 * np.pi * cycles + phase)
    raise ValueError(f"Unsupported input kind: {kind}")


def simulate_trajectory(
    cfg: dict,
    scenario: Scenario,
    input_kind: str,
    amplitude: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    burnin = int(cfg["simulation"]["burnin"])
    n_eff = int(cfg["simulation"]["effective_length"])
    max_diag = int(cfg["model"]["diagnostic_max_lag"])
    total = burnin + max_diag + n_eff + 1
    u = make_input(input_kind, amplitude, total, rng)
    x = np.zeros(total + 1, dtype=float)
    y = np.zeros(total + 1, dtype=float)
    a = float(cfg["simulation"]["a"])
    sigma_process = float(cfg["simulation"]["sigma_process"])
    sigma_observation = float(cfg["simulation"]["sigma_observation"])

    eps = rng.normal(0.0, sigma_process, size=total + 1)
    eta = rng.normal(0.0, sigma_observation, size=total + 1)

    for t in range(max_diag, total):
        history = 0.0
        for lag, coef in scenario.true_lags.items():
            history += coef * x[t - lag]
        x[t + 1] = a * x[t] + u[t] + history + eps[t]
        y[t + 1] = x[t + 1] + eta[t + 1]

    start = burnin + max_diag
    stop = start + n_eff
    return {
        "x": x[start - max_diag : stop + 1].copy(),
        "u": u[start - max_diag : stop + 1].copy(),
        "y": y[start - max_diag : stop + 1].copy(),
        "offset": np.array([max_diag], dtype=int),
    }


def trajectory_hash(traj: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("x", "u", "y"):
        h.update(np.ascontiguousarray(traj[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def validation_schedule(cfg: dict, count: int, rng: np.random.Generator) -> list[tuple[str, float]]:
    kinds = cfg["interventions"]["train_kinds"]
    amplitudes = cfg["interventions"]["train_amplitudes"]
    cells = [(str(k), float(a)) for k in kinds for a in amplitudes]
    schedule = [cells[i % len(cells)] for i in range(count)]
    rng.shuffle(schedule)
    return schedule


def test_schedule(cfg: dict) -> list[tuple[str, float]]:
    per_cell = int(cfg["sample_sizes"]["test_per_cell"])
    return [
        (str(kind), float(amp))
        for kind in cfg["interventions"]["test_kinds"]
        for amp in cfg["interventions"]["test_amplitudes"]
        for _ in range(per_cell)
    ]


def generate_split(
    cfg: dict,
    scenario: Scenario,
    master_seed: int,
    scenario_index: int,
    split_name: str,
    split_index: int,
    count: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, str | int | float]]]:
    schedule_rng = np.random.default_rng(seed_for(master_seed, scenario_index, split_index, 10_000_000))
    schedule = test_schedule(cfg) if split_name == "test" else validation_schedule(cfg, count, schedule_rng)
    if len(schedule) != count:
        raise RuntimeError(f"{split_name}: expected {count} schedules, got {len(schedule)}")

    trajectories = []
    hashes = []
    for idx, (kind, amp) in enumerate(schedule):
        rng = np.random.default_rng(seed_for(master_seed, scenario_index, split_index, idx))
        traj = simulate_trajectory(cfg, scenario, kind, amp, rng)
        trajectories.append(traj)
        hashes.append(
            {
                "master_seed": master_seed,
                "scenario_id": scenario.public_id,
                "split": split_name,
                "trajectory_index": idx,
                "input_kind": kind,
                "amplitude": amp,
                "sha256": trajectory_hash(traj),
            }
        )
    return trajectories, hashes


def row_arrays(
    trajectories: Sequence[dict[str, np.ndarray]],
    x_lags: Sequence[int],
    include_u_lags: bool = False,
    max_u_lag: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = []
    targets = []
    trajectory_ids = []
    for traj_idx, traj in enumerate(trajectories):
        x = traj["x"]
        u = traj["u"]
        y = traj["y"]
        offset = int(traj["offset"][0])
        n_eff = len(y) - offset - 1
        t_index = np.arange(offset, offset + n_eff, dtype=int)

        cols = [
            np.ones(n_eff, dtype=float),
            x[t_index],
            u[t_index],
        ]
        for lag in x_lags:
            cols.append(x[t_index - int(lag)])
        if include_u_lags:
            for lag in range(1, max_u_lag + 1):
                cols.append(u[t_index - lag])
        X = np.column_stack(cols)
        features.append(X)
        targets.append(y[t_index + 1])
        trajectory_ids.append(np.full(n_eff, traj_idx, dtype=int))
    return np.vstack(features), np.concatenate(targets), np.concatenate(trajectory_ids)


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = X.T @ X
    xty = X.T @ y
    penalty = np.eye(X.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, xty)


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - pred))))


def candidate_structures(max_lag: int) -> list[tuple[int, ...]]:
    candidates: list[tuple[int, ...]] = [tuple()]
    candidates.extend((lag,) for lag in range(1, max_lag + 1))
    candidates.extend(tuple(c) for c in combinations(range(1, max_lag + 1), 2))
    return candidates


def select_alpha(
    train_traj: Sequence[dict[str, np.ndarray]],
    inner_traj: Sequence[dict[str, np.ndarray]],
    x_lags: Sequence[int],
    alphas: Sequence[float],
    include_u_lags: bool = False,
    max_u_lag: int = 0,
) -> tuple[float, float]:
    X_train, y_train, _ = row_arrays(train_traj, x_lags, include_u_lags, max_u_lag)
    X_inner, y_inner, _ = row_arrays(inner_traj, x_lags, include_u_lags, max_u_lag)
    best = (float("inf"), float(alphas[0]))
    for alpha in alphas:
        beta = fit_ridge(X_train, y_train, float(alpha))
        score = rmse(y_inner, X_inner @ beta)
        candidate = (score, float(alpha))
        if candidate < best:
            best = candidate
    return best[1], best[0]


def fit_selected_structure(
    cfg: dict,
    train_traj: Sequence[dict[str, np.ndarray]],
    inner_traj: Sequence[dict[str, np.ndarray]],
    outer_traj: Sequence[dict[str, np.ndarray]],
) -> tuple[tuple[int, ...], float, np.ndarray, dict[str, float]]:
    max_lag = int(cfg["model"]["candidate_max_lag"])
    alphas = [float(a) for a in cfg["model"]["alphas"]]
    tolerance = float(cfg["model"]["parsimony_relative_tolerance"])
    records = []

    for lags in candidate_structures(max_lag):
        alpha, inner_score = select_alpha(train_traj, inner_traj, lags, alphas)
        fit_data = list(train_traj) + list(inner_traj)
        X_fit, y_fit, _ = row_arrays(fit_data, lags)
        beta = fit_ridge(X_fit, y_fit, alpha)
        X_outer, y_outer, _ = row_arrays(outer_traj, lags)
        outer_score = rmse(y_outer, X_outer @ beta)
        records.append(
            {
                "lags": tuple(lags),
                "alpha": alpha,
                "inner_rmse": inner_score,
                "outer_rmse": outer_score,
            }
        )

    best_rmse = min(r["outer_rmse"] for r in records)
    eligible = [r for r in records if r["outer_rmse"] <= best_rmse * (1.0 + tolerance)]
    chosen = sorted(eligible, key=lambda r: (len(r["lags"]), r["outer_rmse"], r["lags"]))[0]

    all_fit = list(train_traj) + list(inner_traj) + list(outer_traj)
    X_all, y_all, _ = row_arrays(all_fit, chosen["lags"])
    beta = fit_ridge(X_all, y_all, chosen["alpha"])
    diagnostics = {
        "selected_inner_rmse": float(chosen["inner_rmse"]),
        "selected_outer_rmse": float(chosen["outer_rmse"]),
        "best_outer_rmse": float(best_rmse),
    }
    return tuple(chosen["lags"]), float(chosen["alpha"]), beta, diagnostics


def fit_delay_arx(
    cfg: dict,
    train_traj: Sequence[dict[str, np.ndarray]],
    inner_traj: Sequence[dict[str, np.ndarray]],
    outer_traj: Sequence[dict[str, np.ndarray]],
) -> tuple[float, np.ndarray]:
    max_lag = int(cfg["model"]["candidate_max_lag"])
    alphas = [float(a) for a in cfg["model"]["alphas"]]
    x_lags = tuple(range(1, max_lag + 1))
    alpha, _ = select_alpha(
        train_traj,
        inner_traj,
        x_lags,
        alphas,
        include_u_lags=True,
        max_u_lag=max_lag,
    )
    all_fit = list(train_traj) + list(inner_traj) + list(outer_traj)
    X_all, y_all, _ = row_arrays(
        all_fit,
        x_lags,
        include_u_lags=True,
        max_u_lag=max_lag,
    )
    beta = fit_ridge(X_all, y_all, alpha)
    return alpha, beta


def conditional_shuffle(
    X: np.ndarray,
    history_start_col: int,
    rng: np.random.Generator,
    bins: int,
) -> np.ndarray:
    shuffled = X.copy()
    if X.shape[1] <= history_start_col:
        return shuffled
    x_now = X[:, 1]
    u_now = X[:, 2]
    qx = np.unique(np.quantile(x_now, np.linspace(0.0, 1.0, bins + 1)))
    qu = np.unique(np.quantile(u_now, np.linspace(0.0, 1.0, bins + 1)))
    if len(qx) < 3 or len(qu) < 3:
        order = rng.permutation(len(X))
        shuffled[:, history_start_col:] = X[order, history_start_col:]
        return shuffled
    bx = np.clip(np.digitize(x_now, qx[1:-1]), 0, len(qx) - 2)
    bu = np.clip(np.digitize(u_now, qu[1:-1]), 0, len(qu) - 2)
    group = bx * (len(qu) - 1) + bu
    for g in np.unique(group):
        idx = np.flatnonzero(group == g)
        if len(idx) > 1:
            perm = rng.permutation(idx)
            shuffled[idx, history_start_col:] = X[perm, history_start_col:]
    return shuffled


def residual_misspecification_test(
    cfg: dict,
    diagnostic_traj: Sequence[dict[str, np.ndarray]],
    selected_lags: Sequence[int],
    beta: np.ndarray,
    rng: np.random.Generator,
) -> tuple[bool, float, float, int]:
    candidate_max = int(cfg["model"]["candidate_max_lag"])
    diagnostic_max = int(cfg["model"]["diagnostic_max_lag"])
    permutation_reps = int(cfg["diagnostic"]["permutation_reps"])
    alpha = float(cfg["diagnostic"]["alpha"])

    X_selected, y, trajectory_ids = row_arrays(diagnostic_traj, selected_lags)
    residual = y - X_selected @ beta

    lag_arrays = []
    for lag in range(candidate_max + 1, diagnostic_max + 1):
        X_lag, _, _ = row_arrays(diagnostic_traj, (lag,))
        lag_arrays.append(X_lag[:, 3])
    Z = np.column_stack(lag_arrays)

    def max_abs_corr(a: np.ndarray, z: np.ndarray) -> tuple[float, int]:
        a_center = a - np.mean(a)
        a_norm = np.linalg.norm(a_center)
        best_value = -1.0
        best_index = 0
        for j in range(z.shape[1]):
            zc = z[:, j] - np.mean(z[:, j])
            denom = a_norm * np.linalg.norm(zc)
            value = 0.0 if denom == 0.0 else abs(float(a_center @ zc / denom))
            if value > best_value:
                best_value = value
                best_index = j
        return best_value, best_index

    observed, best_index = max_abs_corr(residual, Z)
    unique_ids = np.unique(trajectory_ids)
    null_values = np.empty(permutation_reps, dtype=float)
    for rep in range(permutation_reps):
        permuted_ids = rng.permutation(unique_ids)
        mapping = dict(zip(unique_ids.tolist(), permuted_ids.tolist()))
        Zp = np.empty_like(Z)
        for source_id in unique_ids:
            target_id = mapping[int(source_id)]
            src_rows = np.flatnonzero(trajectory_ids == source_id)
            target_rows = np.flatnonzero(trajectory_ids == target_id)
            Zp[src_rows, :] = Z[target_rows, :]
        null_values[rep], _ = max_abs_corr(residual, Zp)
    threshold = float(np.quantile(null_values, 1.0 - alpha, method="higher"))
    detected = bool(observed > threshold)
    best_lag = candidate_max + 1 + best_index
    return detected, observed, threshold, best_lag


def evaluate_one(
    cfg: dict,
    scenario: Scenario,
    master_seed: int,
    scenario_index: int,
    output_rows: list[dict],
    selection_rows: list[dict],
    hash_rows: list[dict],
) -> None:
    sizes = cfg["sample_sizes"]
    split_specs = [
        ("train", 1, int(sizes["train"])),
        ("inner", 2, int(sizes["inner"])),
        ("outer", 3, int(sizes["outer"])),
        ("diagnostic", 4, int(sizes["diagnostic"])),
        ("test", 5, len(test_schedule(cfg))),
    ]
    splits: dict[str, list[dict[str, np.ndarray]]] = {}
    for split_name, split_index, count in split_specs:
        trajectories, hashes = generate_split(
            cfg,
            scenario,
            master_seed,
            scenario_index,
            split_name,
            split_index,
            count,
        )
        splits[split_name] = trajectories
        hash_rows.extend(hashes)

    selected_lags, selected_alpha, beta, selection_diag = fit_selected_structure(
        cfg, splits["train"], splits["inner"], splits["outer"]
    )
    delay_alpha, delay_beta = fit_delay_arx(
        cfg, splits["train"], splits["inner"], splits["outer"]
    )

    X_test, y_test, _ = row_arrays(splits["test"], selected_lags)
    pred_adaptive = X_test @ beta
    adaptive_rmse = rmse(y_test, pred_adaptive)

    max_lag = int(cfg["model"]["candidate_max_lag"])
    X_delay, y_delay, _ = row_arrays(
        splits["test"],
        tuple(range(1, max_lag + 1)),
        include_u_lags=True,
        max_u_lag=max_lag,
    )
    delay_rmse = rmse(y_delay, X_delay @ delay_beta)

    shuffle_rng = np.random.default_rng(seed_for(master_seed, scenario_index, 6, 0))
    X_shuffled = conditional_shuffle(
        X_test,
        history_start_col=3,
        rng=shuffle_rng,
        bins=int(cfg["shuffle"]["conditional_bins"]),
    )
    shuffled_rmse = rmse(y_test, X_shuffled @ beta)

    diag_rng = np.random.default_rng(seed_for(master_seed, scenario_index, 7, 0))
    misspec_detected, residual_corr, residual_threshold, best_diag_lag = residual_misspecification_test(
        cfg, splits["diagnostic"], selected_lags, beta, diag_rng
    )

    finite_selected_lags, finite_alpha, finite_beta, _ = fit_selected_structure(
        cfg,
        splits["train"][: int(sizes["finite_train"])],
        splits["inner"],
        splits["outer"],
    )
    finite_delay_alpha, finite_delay_beta = fit_delay_arx(
        cfg,
        splits["train"][: int(sizes["finite_train"])],
        splits["inner"],
        splits["outer"],
    )
    X_finite, y_finite, _ = row_arrays(splits["test"], finite_selected_lags)
    finite_adaptive_rmse = rmse(y_finite, X_finite @ finite_beta)
    X_finite_delay, y_finite_delay, _ = row_arrays(
        splits["test"],
        tuple(range(1, max_lag + 1)),
        include_u_lags=True,
        max_u_lag=max_lag,
    )
    finite_delay_rmse = rmse(y_finite_delay, X_finite_delay @ finite_delay_beta)

    selection_rows.append(
        {
            "master_seed": master_seed,
            "scenario_id": scenario.public_id,
            "selected_lags": ",".join(map(str, selected_lags)),
            "selected_lag_count": len(selected_lags),
            "selected_alpha": selected_alpha,
            "finite_selected_lags": ",".join(map(str, finite_selected_lags)),
            "finite_selected_lag_count": len(finite_selected_lags),
            "finite_selected_alpha": finite_alpha,
            **selection_diag,
        }
    )

    output_rows.append(
        {
            "master_seed": master_seed,
            "scenario_id": scenario.public_id,
            "adaptive_rmse": adaptive_rmse,
            "delay_arx_rmse": delay_rmse,
            "adaptive_delay_ratio": adaptive_rmse / delay_rmse,
            "shuffled_rmse": shuffled_rmse,
            "shuffle_ordered_ratio": shuffled_rmse / adaptive_rmse,
            "misspec_detected": int(misspec_detected),
            "residual_max_abs_corr": residual_corr,
            "residual_null_threshold": residual_threshold,
            "best_diagnostic_lag": best_diag_lag,
            "finite_adaptive_rmse": finite_adaptive_rmse,
            "finite_delay_arx_rmse": finite_delay_rmse,
            "finite_adaptive_delay_ratio": finite_adaptive_rmse / finite_delay_rmse,
            "spectral_radius": stable_radius(float(cfg["simulation"]["a"]), scenario.true_lags),
            "delay_alpha": delay_alpha,
            "finite_delay_alpha": finite_delay_alpha,
        }
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(mode: str, max_seeds: int | None = None) -> None:
    cfg = load_config()
    scenarios = scenario_table(cfg)
    all_seeds = load_seeds()
    if mode == "smoke":
        seeds = all_seeds[:1]
        scenarios = scenarios[:2]
    else:
        seeds = all_seeds if max_seeds is None else all_seeds[:max_seeds]

    out_root = ROOT / "results" / ("smoke" if mode == "smoke" else "r2a")
    out_root.mkdir(parents=True, exist_ok=True)

    metrics: list[dict] = []
    selections: list[dict] = []
    hashes: list[dict] = []
    for seed_idx, seed in enumerate(seeds):
        for scenario_index, scenario in enumerate(scenarios):
            print(f"[r2a] seed {seed_idx + 1}/{len(seeds)} scenario={scenario.public_id}", flush=True)
            evaluate_one(cfg, scenario, seed, scenario_index, metrics, selections, hashes)

    write_csv(out_root / "metrics.csv", metrics)
    write_csv(out_root / "selection.csv", selections)
    write_csv(out_root / "trajectory_hashes.csv", hashes)

    truth_key = {
        "experiment_id": cfg["experiment_id"],
        "mode": mode,
        "scenario_truth": {s.public_id: sorted(s.true_lags) for s in scenarios},
        "seed_manifest_sha256": hashlib.sha256(
            json.dumps(seeds, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "row_counts": {
            "metrics": len(metrics),
            "selection": len(selections),
            "trajectory_hashes": len(hashes),
        },
    }
    with (out_root / "truth_key.json").open("w", encoding="utf-8") as f:
        json.dump(truth_key, f, ensure_ascii=False, indent=2, sort_keys=True)

    unique_hashes = {row["sha256"] for row in hashes}
    smoke_payload = {
        "passed": len(unique_hashes) == len(hashes)
        and all(np.isfinite(float(row["adaptive_rmse"])) for row in metrics)
        and all(float(row["spectral_radius"]) < 1.0 for row in metrics),
        "trajectory_count": len(hashes),
        "unique_trajectory_hashes": len(unique_hashes),
        "metric_rows": len(metrics),
    }
    with (out_root / "smoke_result.json").open("w", encoding="utf-8") as f:
        json.dump(smoke_payload, f, ensure_ascii=False, indent=2)
    if not smoke_payload["passed"]:
        raise SystemExit("Smoke/integrity checks failed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "confirm"], default="confirm")
    parser.add_argument("--max-seeds", type=int, default=None)
    args = parser.parse_args()
    run(args.mode, args.max_seeds)


if __name__ == "__main__":
    main()
