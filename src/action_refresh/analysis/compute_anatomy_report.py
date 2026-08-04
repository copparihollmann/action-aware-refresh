"""Aggregate B0–B4 profile records into docs/compute_anatomy.md."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(jsonl: Path) -> dict[str, float]:
    lats: list[float] = []
    stages: dict[str, list[float]] = {}
    for line in jsonl.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("network_roundtrip_ms") is not None:
            lats.append(r["network_roundtrip_ms"])
        for k in ("preprocessing_ms", "vision_encode_ms", "context_ms",
                  "denoising_ms", "vision_decode_ms", "postprocess_ms"):
            v = r.get(k)
            if v is not None:
                stages.setdefault(k, []).append(v)
    out: dict[str, float] = {
        "n": float(len(lats)),
        "roundtrip_mean_ms": statistics.fmean(lats) if lats else float("nan"),
        "roundtrip_p50_ms": _percentile(lats, 0.5),
        "roundtrip_p95_ms": _percentile(lats, 0.95),
    }
    for k, vs in stages.items():
        out[f"{k}_mean"] = statistics.fmean(vs)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d = Path(args.profile_dir)
    files = sorted(d.glob("*.jsonl"))
    lines: list[str] = []
    lines.append("# Compute anatomy (M2)\n")
    lines.append(f"Source: `{d}`  ({len(files)} configs)\n")
    lines.append("| config | n | wall p50 ms | wall p95 ms | notes |")
    lines.append("|---|---|---|---|---|")
    for f in files:
        s = summarize(f)
        # Load energy sidecar if present.
        ej = f.with_name(f.stem + ".energy.json")
        energy = json.loads(ej.read_text()) if ej.exists() else {}
        note = f"energy≈{energy.get('energy_j', 'n/a')} J" if energy else ""
        lines.append(
            f"| `{f.stem}` | {int(s['n'])} | {s['roundtrip_p50_ms']:.1f} | "
            f"{s['roundtrip_p95_ms']:.1f} | {note} |"
        )

    lines.append("\n## Per-stage means (ms)\n")
    stage_keys = ("preprocessing_ms", "vision_encode_ms", "context_ms",
                  "denoising_ms", "vision_decode_ms", "postprocess_ms")
    lines.append("| config | " + " | ".join(k.replace("_ms","") for k in stage_keys) + " |")
    lines.append("|---|" + "---|" * len(stage_keys))
    for f in files:
        s = summarize(f)
        cells = " | ".join(f"{s.get(k+'_mean', float('nan')):.2f}" for k in stage_keys)
        lines.append(f"| `{f.stem}` | {cells} |")

    lines.append("\n## Go/no-go\n")
    lines.append(
        "Fill in one paragraph after inspecting the table above:\n"
        "1. What fraction of end-to-end time is *shared/action* vs *vision*?\n"
        "2. Does disabling `decode_video` visibly change denoising time?\n"
        "3. Do action-relevant token counts match what we expect from the model card?\n"
        "\n"
        "**Decision:** if vision cost is <20% of end-to-end, we deprioritize\n"
        "the visual-imagination branch (M6, M7) and shift to action-only\n"
        "policy-invocation scheduling (M3, M4)."
    )

    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
