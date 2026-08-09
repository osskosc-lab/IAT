"""Independent artifact validator for a completed Phase 3A analysis."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result", default="results/phase3a/analysis.json")
    p.add_argument("--predictions", default="results/phase3a/predictions.csv")
    p.add_argument("--data", default="data/phase3a/trials.csv")
    p.add_argument("--spec", default="config/phase3a_spec.yaml")
    args = p.parse_args()
    report = json.loads(Path(args.result).read_text(encoding="utf-8"))
    with open(args.predictions, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rmse0 = math.sqrt(sum(float(r["sqerr_m0"]) for r in rows) / len(rows))
    rmseh = math.sqrt(sum(float(r["sqerr_mh"]) for r in rows) / len(rows))
    ratio = rmseh / rmse0
    checks = {
        "spec_hash": report["hashes"]["spec"] == sha256(Path(args.spec)),
        "data_hash": report["hashes"]["data"] == sha256(Path(args.data)),
        "rmse_m0": abs(rmse0 - report["primary"]["rmse_m0"]) < 1e-12,
        "rmse_mh": abs(rmseh - report["primary"]["rmse_mh"]) < 1e-12,
        "ratio": abs(ratio - report["primary"]["ratio_mh_m0"]["estimate"]) < 1e-12,
        "device_fold_exclusivity": True,
    }
    folds: dict[str, set[str]] = {}
    for r in rows:
        folds.setdefault(r["device_id"], set()).add(r["fold"])
    checks["device_fold_exclusivity"] = all(len(v) == 1 for v in folds.values())
    ok = all(checks.values())
    out = {"status": "PASS" if ok else "FAIL", "checks": checks}
    Path(args.result).with_name("validator.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
