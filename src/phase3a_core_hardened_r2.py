"""Phase 3A hardened core r2.

Round-1 quadratic hardening was itself falsified by more complex but entirely
observable current-state response functions (smooth oscillatory and threshold
responses).  r2 keeps the r1 policy/audit machinery but replaces the current-
state feature map with a fixed physically-scaled hinge-spline basis available
identically to M0 and MH.

The spline basis is frozen against the provisional observable-equivalence widths.
If the hardware pilot changes those widths, this file must be updated and hashed
before CONFIRMATORY_LOCKED.
"""
from __future__ import annotations

import numpy as np

import phase3a_core_hardened as _r1
from phase3a_core_hardened import *  # re-export the audited r1 API

# Provisional physical scaling constants from config/phase3a_spec.yaml.
# These are deliberately not learned from outcomes.
_V_SCALE = 1.0e-3
_I_FRACTION_SCALE = 1.0e-3
_SLOPE_SCALE = 2.0e-4
_TEMP_SCALE = 2.0e-1
_SPLINE_KNOTS = np.linspace(-0.8, 0.8, 17)


def _hinges(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z[:, None] - _SPLINE_KNOTS[None, :])


def _base_matrix_r2(rows: list[dict]) -> np.ndarray:
    pre_v = np.array([_r1._to_float(r, "pre_voltage_v") for r in rows], dtype=float)
    pre_i = np.array([_r1._to_float(r, "pre_current_a") for r in rows], dtype=float)
    pre_i_frac = np.array([_r1._to_float(r, "pre_current_fraction_full_scale") for r in rows], dtype=float)
    slope = np.array([_r1._to_float(r, "pre_voltage_slope_v_per_s") for r in rows], dtype=float)
    temp = np.array([_r1._to_float(r, "pre_temperature_delta_c") for r in rows], dtype=float)
    total = np.array([_r1._to_float(r, "total_stimulus_abs") for r in rows], dtype=float)
    work = np.array([_r1._to_float(r, "conditioning_work_j") for r in rows], dtype=float)
    run = np.array([_r1._to_float(r, "run_index") for r in rows], dtype=float)

    # Normalize the four observable-equivalence coordinates by their physical gate
    # widths.  This avoids fitting spline locations from outcome data.
    zv = pre_v / _V_SCALE
    zi = pre_i_frac / _I_FRACTION_SCALE
    zs = slope / _SLOPE_SCALE
    zt = temp / _TEMP_SCALE

    linear = np.column_stack([
        pre_v, pre_i, pre_i_frac, slope, temp, total, work, run,
    ])
    spline = np.column_stack([
        _hinges(zv), _hinges(zi), _hinges(zs), _hinges(zt),
    ])
    interactions = np.column_stack([
        zv * zi,
        zv * zt,
        zi * zt,
        zv * zs,
        zi * zs,
        zs * zt,
        pre_v * total,
        pre_i * total,
        temp * work,
        run**2,
    ])
    last = np.array(
        [[float(r["last_stimulus"] == x) for x in ["A", "B", "C"]] for r in rows],
        dtype=float,
    )
    block_levels = sorted(set(r["block_id"] for r in rows))
    block = np.array(
        [[float(r["block_id"] == x) for x in block_levels] for r in rows],
        dtype=float,
    )
    return np.column_stack([linear, spline, interactions, last, block])


# Functions defined in the r1 module resolve globals in the r1 module at runtime.
# Replacing this private feature-map function therefore upgrades build_matrix,
# cross_validated_predictions, analyze_subset, etc., while preserving all r1
# gate, shuffle, bootstrap, and audit code for traceability.
_r1._base_matrix = _base_matrix_r2

# Explicit aliases make static readers aware that the public API is the r1 API
# executed with the r2 feature map.
load_spec = _r1.load_spec
sha256_file = _r1.sha256_file
read_csv = _r1.read_csv
write_csv = _r1.write_csv
history_features = _r1.history_features
validate_trial_rows = _r1.validate_trial_rows
apply_prestate_policy = _r1.apply_prestate_policy
schedule_audit = _r1.schedule_audit
observable_balance_audit = _r1.observable_balance_audit
build_matrix = _r1.build_matrix
group_folds = _r1.group_folds
cross_validated_predictions = _r1.cross_validated_predictions
bootstrap_ratio = _r1.bootstrap_ratio
conditional_shuffle = _r1.conditional_shuffle
analyze_subset = _r1.analyze_subset
equivalence_pass = _r1.equivalence_pass
gate_status = _r1.gate_status
save_json = _r1.save_json
