from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phase3a_core import group_folds, history_features, load_spec, cross_validated_predictions
from phase3a_null_calibration import generate_null
from phase3a_adversarial_falsification import (
    _cv_ratio,
    nonlinear_observable_confound,
    blocked_run_drift,
    true_order_effect,
)

SPEC = Path(__file__).resolve().parents[1] / "config" / "phase3a_spec.yaml"


def test_history_representation_is_fixed_and_order_sensitive():
    abc = history_features("ABC")
    bac = history_features("BAC")
    assert abc.shape == (9,)
    assert not np.array_equal(abc, bac)
    assert abc[:3].tolist() == [1.0, 2.0, 3.0]
    assert bac[:3].tolist() == [2.0, 1.0, 3.0]


def test_group_folds_never_split_a_device():
    devices = [f"D{i:02d}" for i in range(10) for _ in range(4)]
    folds = group_folds(devices, 5, 123)
    assert set().union(*folds) == set(devices)
    assert sum(len(x) for x in folds) == len(set(devices))
    assert all(a.isdisjoint(b) for i, a in enumerate(folds) for b in folds[i + 1:])


def test_simple_null_generator_has_no_material_history_advantage():
    spec = load_spec(SPEC)
    rows = generate_null(spec, 12345, n_devices=12)
    result = cross_validated_predictions(rows, spec)
    assert 0.90 < result.ratio < 1.10
    by_device = {}
    for r in result.rows:
        by_device.setdefault(r["device_id"], set()).add(r["fold"])
    assert all(len(v) == 1 for v in by_device.values())


def test_legacy_linear_baseline_is_falsified_by_nonlinear_observable_confound():
    rows = nonlinear_observable_confound(0)
    assert _cv_ratio(rows, hardened=False) < 0.90
    assert 0.97 < _cv_ratio(rows, hardened=True) < 1.03


def test_hardened_baseline_blocks_acquisition_drift_proxy():
    rows = blocked_run_drift(1)
    assert _cv_ratio(rows, hardened=False) < 0.90
    assert 0.97 < _cv_ratio(rows, hardened=True) < 1.03


def test_hardening_preserves_true_order_sensitivity():
    rows = true_order_effect(2)
    assert _cv_ratio(rows, hardened=True) < 0.90
