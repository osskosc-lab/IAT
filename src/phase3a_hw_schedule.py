from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def build_schedule(definition_path: str | Path) -> list[dict[str, str]]:
    spec = json.loads(Path(definition_path).read_text(encoding="utf-8"))
    decode = spec["encoding"]
    rows: list[dict[str, str]] = []
    for device_id, bits in sorted(spec["main_sequences"].items()):
        if len(bits) != 120:
            raise ValueError(f"{device_id}: expected 120 main trials")
        run_index = 1
        prev = "NONE"
        for i, bit in enumerate(bits):
            order = decode[bit]
            block_id = 1 + i // 20
            rows.append({
                "device_id": device_id,
                "trial_id": f"{device_id}-M{run_index:03d}",
                "trial_class": "main",
                "block_id": str(block_id),
                "run_index": str(run_index),
                "order": order,
                "previous_trial_order": prev,
            })
            prev = order
            run_index += 1
        device_number = int(device_id[1:])
        long_orders = (["AB", "BA"] * 12) if device_number % 2 else (["BA", "AB"] * 12)
        for j, order in enumerate(long_orders, start=1):
            rows.append({
                "device_id": device_id,
                "trial_id": f"{device_id}-L{j:03d}",
                "trial_class": "long_washout",
                "block_id": str(7 + (j - 1) // 4),
                "run_index": str(run_index),
                "order": order,
                "previous_trial_order": prev,
            })
            prev = order
            run_index += 1
    return rows


def csv_text(rows: list[dict[str, str]]) -> str:
    import io
    out = io.StringIO()
    fields = ["device_id", "trial_id", "trial_class", "block_id", "run_index", "order", "previous_trial_order"]
    w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return out.getvalue()


def schedule_sha256(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256(csv_text(rows).encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--definition", default="config/phase3a_hw_schedule.json")
    p.add_argument("--out", default="results/phase3a_hw/frozen_schedule.csv")
    args = p.parse_args()
    definition = json.loads(Path(args.definition).read_text(encoding="utf-8"))
    rows = build_schedule(args.definition)
    digest = schedule_sha256(rows)
    expected = definition["generated_schedule_csv_sha256"]
    if digest != expected:
        raise SystemExit(f"schedule digest mismatch: {digest} != {expected}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(csv_text(rows), encoding="utf-8")
    print(f"rows={len(rows)} sha256={digest}")


if __name__ == "__main__":
    main()
