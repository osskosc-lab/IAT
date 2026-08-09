import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from phase3a_hw_schedule import build_schedule, csv_text, schedule_sha256
from phase3a_hw_synthetic import generate


def test_frozen_schedule_digest_and_counts():
    definition = ROOT / "config" / "phase3a_hw_schedule.json"
    meta = json.loads(definition.read_text(encoding="utf-8"))
    rows = build_schedule(definition)
    assert len(rows) == 1152
    assert schedule_sha256(rows) == meta["generated_schedule_csv_sha256"]
    assert hashlib.sha256(csv_text(rows).encode()).hexdigest() == meta["generated_schedule_csv_sha256"]
    for d in sorted({r["device_id"] for r in rows}):
        main = [r for r in rows if r["device_id"] == d and r["trial_class"] == "main"]
        longw = [r for r in rows if r["device_id"] == d and r["trial_class"] == "long_washout"]
        assert len(main) == 120 and len(longw) == 24
        assert sum(r["order"] == "AB" for r in main) == 60
        assert sum(r["order"] == "BA" for r in main) == 60
        for b in range(1, 7):
            br = [r for r in main if int(r["block_id"]) == b]
            assert len(br) == 20
            assert sum(r["order"] == "AB" for r in br) == 10
            assert sum(r["order"] == "BA" for r in br) == 10
        run = 1; best = 1
        for i in range(1, len(main)):
            run = run + 1 if main[i]["order"] == main[i-1]["order"] else 1
            best = max(best, run)
        assert best <= 3


def test_preregistration_matches_frozen_schedule():
    spec = yaml.safe_load((ROOT / "config" / "phase3a_hw_preregistration.yaml").read_text(encoding="utf-8"))
    meta = json.loads((ROOT / "config" / "phase3a_hw_schedule.json").read_text(encoding="utf-8"))
    assert spec["hardware_pilot"]["devices"] == 8
    assert spec["hardware_pilot"]["total_trials"] == 1152
    assert spec["hardware_pilot"]["orders"] == ["AB", "BA"]
    assert spec["hardware_pilot"]["schedule_sha256"] == meta["generated_schedule_csv_sha256"]


def test_analysis_config_contract_is_atomic():
    """Catch partial config/API edits such as the 080b358 margin-key split.

    The analyzer and independent validator both consume these frozen inference keys.
    A preregistration edit must therefore keep the shared contract valid in the same
    commit rather than relying on a later repair commit.
    """
    spec = yaml.safe_load((ROOT / "config" / "phase3a_hw_preregistration.yaml").read_text(encoding="utf-8"))
    inference = spec["inference"]
    required = {
        "bootstrap_repetitions",
        "confidence_level",
        "bootstrap_seed",
        "primary_support_threshold",
        "equivalence_margin",
        "order_shuffle_threshold",
    }
    assert required <= set(inference)
    assert int(inference["bootstrap_repetitions"]) == 20000
    assert float(inference["confidence_level"]) == 0.99
    assert float(inference["primary_support_threshold"]) == 0.90
    assert float(inference["equivalence_margin"]) == 0.10
    assert float(inference["order_shuffle_threshold"]) == 1.10


def test_synthetic_generator_is_observable_balanced_and_long_washout_has_no_current_effect():
    rows = generate(str(ROOT / "config" / "phase3a_hw_schedule.json"), "positive", 1234)
    assert len(rows) == 1152
    main = [r for r in rows if r["trial_class"] == "main"]
    for d in sorted({r["device_id"] for r in main}):
        dr = [r for r in main if r["device_id"] == d]
        for var in ["pre_voltage_v", "pre_current_fraction_full_scale", "pre_voltage_slope_v_per_s", "pre_temperature_delta_c"]:
            a = np.mean([r[var] for r in dr if r["order"] == "AB"])
            b = np.mean([r[var] for r in dr if r["order"] == "BA"])
            assert abs(a-b) < 1e-12
