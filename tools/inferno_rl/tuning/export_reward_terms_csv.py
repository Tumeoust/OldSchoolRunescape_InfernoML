"""
Export logged reward term scalars from TensorBoard event files into a sorted CSV.

Reads tags written by --log-reward-terms:
  raw_reward_terms/ep_sum_mean/<TERM>
  raw_reward_terms/ep_mean_per_tick_mean/<TERM>

Usage:
  python -m tools.inferno_rl.tuning.export_reward_terms_csv --log-dir logs/V20_climb
  python -m tools.inferno_rl.tuning.export_reward_terms_csv --run-dir logs/V20_climb/<run_subdir>
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


_PREFIXES = (
    "raw_reward_terms/ep_sum_mean/",
    "raw_reward_terms/ep_mean_per_tick_mean/",
)


@dataclass(frozen=True)
class TermRow:
    metric: str
    term: str
    value: float
    abs_value: float
    last_step: int
    num_points: int


def _load_run_dir(log_dir: Path, run: Optional[str]) -> tuple[Path, str]:
    """Return (run_dir, run_name). Auto-selects latest by mtime if run is None."""
    if run is not None:
        run_dir = log_dir / run
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, run

    subdirs = [d for d in log_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {log_dir}")
    latest = max(subdirs, key=lambda d: d.stat().st_mtime)
    return latest, latest.name


def _iter_reward_term_tags(tags: Iterable[str]) -> list[str]:
    out: list[str] = []
    for t in tags:
        if t.startswith(_PREFIXES):
            out.append(t)
    return out


def _metric_from_tag(tag: str) -> tuple[str, str]:
    """Return (metric, term) from a tag."""
    for prefix in _PREFIXES:
        if tag.startswith(prefix):
            metric = prefix[len("raw_reward_terms/") : -1]  # ep_sum_mean or ep_mean_per_tick_mean
            term = tag[len(prefix) :]
            return metric, term
    return "unknown", tag


def _load_rows(
    run_dir: Path,
    last_n: int,
    value_mode: str,
    min_points: int,
) -> list[TermRow]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print(
            "ERROR: tensorboard package not found. Install with: pip install tensorboard",
            file=sys.stderr,
        )
        sys.exit(1)

    ea = EventAccumulator(str(run_dir))
    ea.Reload()

    available = set(ea.Tags().get("scalars", []))
    tags = _iter_reward_term_tags(available)
    rows: list[TermRow] = []

    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        events_sorted = sorted(events, key=lambda e: e.step)
        steps = [e.step for e in events_sorted]
        values = [float(e.value) for e in events_sorted]
        if len(values) < min_points:
            continue

        if value_mode == "final":
            v = values[-1]
        elif value_mode == "mean_last_n":
            taken = values[-last_n:] if len(values) >= last_n else values
            v = float(np.mean(taken))
        else:
            raise ValueError(f"Unknown value_mode: {value_mode}")

        metric, term = _metric_from_tag(tag)
        rows.append(
            TermRow(
                metric=metric,
                term=term,
                value=v,
                abs_value=abs(v),
                last_step=int(steps[-1]),
                num_points=len(values),
            )
        )

    rows.sort(key=lambda r: (r.abs_value, r.value), reverse=True)
    return rows


def _write_csv(rows: list[TermRow], out_path: Optional[Path]) -> None:
    out_file = sys.stdout if out_path is None else out_path.open("w", newline="", encoding="utf-8")
    try:
        writer = csv.writer(out_file)
        writer.writerow(["metric", "term", "value", "abs_value", "last_step", "num_points"])
        for r in rows:
            writer.writerow([r.metric, r.term, f"{r.value:.8g}", f"{r.abs_value:.8g}", r.last_step, r.num_points])
    finally:
        if out_path is not None:
            out_file.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Export reward term scalars to CSV (sorted by magnitude)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--log-dir", type=str, help="Directory containing run subdirs (auto-picks latest unless --run set)")
    src.add_argument("--run-dir", type=str, help="Specific run directory containing event files")
    p.add_argument("--run", type=str, default=None, help="Specific run subdir name (only with --log-dir)")
    p.add_argument("--last-n", type=int, default=50, help="Window size for mean_last_n (default: 50)")
    p.add_argument("--value", choices=["final", "mean_last_n"], default="final",
                   help="Value per term: final point or mean over last-n points")
    p.add_argument("--min-points", type=int, default=1, help="Skip tags with fewer points (default: 1)")
    p.add_argument("--top", type=int, default=0, help="Limit to top N rows (0 = all)")
    p.add_argument("--output", type=str, default=None, help="Output .csv path (omit to write to stdout)")
    args = p.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            print(f"ERROR: run-dir not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
        run_name = run_dir.name
    else:
        log_dir = Path(args.log_dir)
        if not log_dir.is_dir():
            print(f"ERROR: log-dir not found: {log_dir}", file=sys.stderr)
            sys.exit(1)
        run_dir, run_name = _load_run_dir(log_dir, args.run)

    rows = _load_rows(
        run_dir=run_dir,
        last_n=max(1, int(args.last_n)),
        value_mode=args.value,
        min_points=max(1, int(args.min_points)),
    )

    if args.top and args.top > 0:
        rows = rows[: int(args.top)]

    out_path = Path(args.output) if args.output else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(
            f"WARNING: no reward-term tags found in {run_dir} (run={run_name}). "
            "Did you train with --log-reward-terms?",
            file=sys.stderr,
        )

    _write_csv(rows, out_path)


if __name__ == "__main__":
    main()

