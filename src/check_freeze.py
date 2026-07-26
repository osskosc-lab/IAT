#!/usr/bin/env python3
"""Fail closed if a preregistered tracked file changed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "frozen_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures = []
    for rel, expected in payload["files"].items():
        path = ROOT / rel
        if not path.exists():
            failures.append(f"missing: {rel}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(f"hash mismatch: {rel}: expected {expected}, got {actual}")
    if failures:
        raise SystemExit("Freeze verification failed:\n" + "\n".join(failures))
    print(f"Freeze verification PASS ({len(payload['files'])} files)")


if __name__ == "__main__":
    main()
