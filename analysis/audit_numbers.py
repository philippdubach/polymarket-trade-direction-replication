"""Trace every number in the manuscript to a JSON result.

Extracts `\\SI{x}{...}`, `\\num{x}`, comma-grouped integers (`1{,}217{,}156`) and
bare decimals from the tex, skips years and reference labels, and looks each
one up in `claims.json`:

    {"92.9": {"json": "a1_fill_match_2026-05-13.json", "path": "coverage.print_matched_share",
              "scale": 100, "tol": 0.05, "note": "coverage on 13 May"}, ...}

keyed by the number's text as it appears in the tex, or by `<text>@<file>` when the same
text carries another meaning in another file. A number with no claim is
unsourced; a claim whose JSON value, scaled, differs by more than `tol` is a
mismatch. The audit is done when there are zero unsourced numbers and zero
mismatches.

Run:  uv run python audit_numbers.py /path/to/manuscript.tex [claims.json]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from d0_common import HERE

NUM = r"-?\d+(?:\.\d+)?"
PATTERNS = [
    (re.compile(r"\\SI\{(" + NUM + r")\}\{[^}]*\}"), "si"),
    (re.compile(r"\\num\{(" + NUM + r"|\d{1,3}(?:\{,\}\d{3})+)\}"), "num"),
    (re.compile(r"(?<![\w.])(\d{1,3}(?:\{,\}\d{3})+)(?![\w.])"), "grouped"),
    (re.compile(r"(?<![\w.{\-])(-?\d+\.\d+)(?![\w}])"), "decimal"),
]
YEAR = re.compile(r"^(19|20)\d{2}$")
SKIP_CONTEXT = re.compile(r"\\(ref|label|cite[pt]?|includegraphics|input|bibliography|section|subsection|caption|url|href)(\[[^\]]*\])?\{[^}]*\}")


def _to_value(text: str) -> float:
    return float(text.replace("{,}", "").replace(",", ""))


def extract(tex: str) -> list[dict]:
    out, seen = [], set()
    for lineno, raw in enumerate(tex.splitlines(), 1):
        line = SKIP_CONTEXT.sub("", raw)
        if line.lstrip().startswith("%"):
            continue
        for pat, kind in PATTERNS:
            for m in pat.finditer(line):
                text = m.group(1)
                key = (lineno, m.start(1), text)
                if key in seen:
                    continue
                if kind == "decimal" and YEAR.match(text):
                    continue
                seen.add(key)
                out.append({"text": text, "value": _to_value(text), "kind": kind, "line": lineno})
    return out


def _split_path(path: str) -> list:
    """`a.b["k=1.5"][0].c` -> ['a', 'b', 'k=1.5', 0, 'c']: dots separate, quoted brackets keep dots."""
    parts, i = [], 0
    while i < len(path):
        if path[i] == ".":
            i += 1
        elif path[i] == "[":
            j = path.index("]", i)
            inner = path[i + 1:j]
            parts.append(inner.strip('"') if inner.startswith('"') else int(inner))
            i = j + 1
        else:
            j = i
            while j < len(path) and path[j] not in ".[":
                j += 1
            parts.append(path[i:j])
            i = j
    return parts


def lookup(data, path: str):
    cur = data
    for part in _split_path(path):
        cur = cur[part]
    return cur


def match(found: list[dict], claims: dict, store: dict, filename: str | None = None) -> list[dict]:
    """A claim keyed `<text>@<file>` takes precedence in that file over the bare `<text>` key, so a
    number used with two meanings in two files is checked against the right source in each."""
    rep = []
    for f in found:
        c = (claims.get(f"{f['text']}@{filename}") if filename else None) or claims.get(f["text"])
        if not c:
            rep.append(f | {"status": "unsourced"})
            continue
        if "constant" in c:
            rep.append(f | {"status": "constant", "claim": c})
            continue
        try:
            v = lookup(store[c["json"]], c["path"]) * c.get("scale", 1)
        except (KeyError, IndexError, TypeError) as e:
            rep.append(f | {"status": "missing_json", "error": str(e)})
            continue
        tol = c.get("tol", 0.0005)
        rep.append(f | {"status": "ok" if abs(v - f["value"]) <= tol else "mismatch", "json_value": v, "claim": c})
    return rep


def summary(rep: list[dict]) -> dict:
    return {"total": len(rep),
            "ok": sum(1 for r in rep if r["status"] in ("ok", "constant")),
            "mismatch": sum(1 for r in rep if r["status"] == "mismatch"),
            "missing_json": sum(1 for r in rep if r["status"] == "missing_json"),
            "unsourced": [r["text"] for r in rep if r["status"] == "unsourced"]}


def main() -> None:
    tex = Path(sys.argv[1]).read_text()
    claims_path = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "claims.json"
    claims = json.loads(claims_path.read_text()) if claims_path.exists() else {}
    store = {p.name: json.loads(p.read_text()) for p in HERE.glob("*.json") if p.name not in ("claims.json", "manifest.json")}
    rep = match(extract(tex), claims, store, filename=Path(sys.argv[1]).name)
    s = summary(rep)
    for r in rep:
        if r["status"] not in ("ok", "constant"):
            print(f"  line {r['line']:>4}  {r['status']:<12} {r['text']}" + (f"  json={r['json_value']}" if "json_value" in r else ""))
    print(f"total {s['total']}  ok {s['ok']}  mismatch {s['mismatch']}  missing_json {s['missing_json']}  unsourced {len(s['unsourced'])}")
    raise SystemExit(0 if not s["unsourced"] and not s["mismatch"] and not s["missing_json"] else 1)


if __name__ == "__main__":
    main()
