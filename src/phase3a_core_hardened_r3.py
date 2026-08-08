"""Phase 3A hardened core r3: explicit previous-trial carryover adjustment.

Round 3 targets a different failure mode from current-trial nonlinear state:
residual memory from the *previous experimental trial*.  If the acquisition
schedule links previous and current order, current order labels can predict a
carryover artifact even when the current A/B/C sequence has no effect.

r3 therefore adds previous-trial order and measured reset elapsed time to the
observable-current-state baseline M0 and, identically, to MH.  The r2 spline
basis remains unchanged.

This is PILOT_OPEN code; hardware reset adequacy must still be established in
the eight-device pilot.
"""
from __future__ import annotations

import numpy as np

import phase3a_core_hardened_r2 as _r2
import phase3a_core_hardened as _r1
from phase3a_core_hardened_r2 import *

_PREV_LEVELS = ["NONE", "ABC", "ACB", "BAC", "BCA", "CAB", "CBA"]


def _base_matrix_r3(rows: list[dict]) -> np.ndarray:
    # Build the already-hardened current-trial r2 representation first.
    x = _r2._base_matrix_r2(rows)
    prev = np.array(
        [[float(r.get("previous_trial_order", "NONE") == level) for level in _PREV_LEVELS]
         for r in rows],
        dtype=float,
    )
    reset = np.array(
        [float(r.get("reset_elapsed_s", 0.0) or 0.0) for r in rows],
        dtype=float,
    )
    return np.column_stack([x, prev, reset, reset**2])


# r1 owns build_matrix/cross-validation globals.  Point its feature map to r3.
_r1._base_matrix = _base_matrix_r3

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
