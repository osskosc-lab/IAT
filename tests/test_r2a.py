from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r2a_experiment import candidate_structures, load_config, stable_radius


def test_candidate_count():
    assert len(candidate_structures(6)) == 22


def test_all_generators_are_stable():
    cfg = load_config()
    a = float(cfg["simulation"]["a"])
    for scenario in cfg["scenarios"]:
        lags = {int(k): float(v) for k, v in scenario["true_lags"].items()}
        assert stable_radius(a, lags) < 1.0


def test_seed_manifest_unique():
    path = ROOT / "config" / "r2a_confirmatory_seeds.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    seeds = payload["seeds"]
    assert payload["count"] == 30
    assert len(seeds) == len(set(seeds)) == 30
