"""Reproducibility manifest: every input the scripts read and every output they wrote.

Archive hours reuse the downloader's receipt (sha256, rows, bytes) rather than
re-hashing 1.7 TB; V1 hours, which have no receipts, get size and mtime plus a
full hash when --hash-v1 is passed. Everything derived under
/Volumes/data/paper-b-ext and every JSON here is hashed. Key material is refused.

Run:  uv run python d5_manifest.py [--hash-v1]
Out:  manifest.json (here)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from d0_common import EXT, HERE, REPO, V1_BASE, V2_BASE, all_days, day_files, git_head

KEY_MARKERS = ("alchemy.key", ".key", "secret", "token.txt")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def entry(p: Path, root: Path) -> dict:
    if any(m in p.name.lower() for m in KEY_MARKERS):
        raise ValueError(f"refusing to list key material: {p.name}")
    return {"path": str(p.relative_to(root)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def archive_entry(p: Path, hash_missing: bool = False) -> dict:
    # The downloader appends the suffix to the complete Parquet filename.
    r = p.with_name(p.name + ".receipt.json")
    if not r.exists():
        r = p.with_suffix(".receipt.json")  # support legacy receipt naming
    if r.exists():
        d = json.loads(r.read_text())
        return {"path": p.name, "bytes": d.get("bytes", p.stat().st_size), "rows": d.get("rows"), "sha256": d.get("sha256"), "source": "receipt"}
    e = {"path": p.name, "bytes": p.stat().st_size, "mtime": int(p.stat().st_mtime), "source": "stat"}
    if hash_missing:
        e["sha256"] = sha256_file(p)
        e["source"] = "hashed"
    return e


def walk(root: Path) -> list[dict]:
    """Every regular, non-hidden file under root, relative to it. Key material is skipped, not listed."""
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.name.startswith(".") and not any(m in p.name.lower() for m in KEY_MARKERS):
            out.append(entry(p, root))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash-v1", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="Refresh archive receipts and code/result hashes; preserve previously hashed derived data")
    a = ap.parse_args()
    prior = json.loads((HERE / "manifest.json").read_text()) if a.refresh else None
    if prior:
        v2 = [archive_entry(Path(V2_BASE) / e["path"]) for e in prior["archive_v2"]["files"]]
        v1 = prior["archive_v1"]["files"]
        derived = prior["derived"]["entries"]
    else:
        v2 = [archive_entry(Path(f)) for d in all_days("v2") for f in day_files(d, "v2")]
        v1 = [archive_entry(Path(f), hash_missing=a.hash_v1) for d in all_days("v1") for f in day_files(d, "v1")]
        derived = walk(EXT)
    results = [entry(p, HERE) for p in sorted(HERE.glob("*.json")) if p.name != "manifest.json"]
    scripts = [entry(p, HERE) for p in sorted(HERE.glob("*.py"))]
    try:
        lock = subprocess.run(["uv", "lock", "--check"], cwd=REPO, capture_output=True, text=True, timeout=120)
        lock_ok = lock.returncode == 0
    except Exception:  # noqa: BLE001
        lock_ok = None
    man = {
        "git_head": git_head(),
        "derived_hashes_reused": bool(prior),
        "uv_lock_check_ok": lock_ok,
        "uv_lock_sha256": sha256_file(REPO / "uv.lock") if (REPO / "uv.lock").exists() else None,
        "archive_v2": {"base": V2_BASE, "hours": len(v2), "rows": sum(e.get("rows") or 0 for e in v2), "bytes": sum(e["bytes"] for e in v2), "files": v2},
        "archive_v1": {"base": V1_BASE, "hours": len(v1), "bytes": sum(e["bytes"] for e in v1), "files": v1},
        "derived": {"root": str(EXT), "files": len(derived), "bytes": sum(e["bytes"] for e in derived), "entries": derived},
        "results": results,
        "scripts": scripts,
    }
    (HERE / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"manifest: v2 {len(v2)} hours, v1 {len(v1)} hours, derived {len(derived)} files, results {len(results)}, scripts {len(scripts)}")


if __name__ == "__main__":
    main()
