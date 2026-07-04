"""
Reads TensorBoard event files and produces a Markdown summary of training metrics.

Usage:
    python -m tools.inferno_rl.tuning.tb_summary
      --log-dir   tools/inferno_rl/logs/inferno_gpu
      --run       <run_name>           # optional; uses latest subdir if omitted
      --last-n    50                   # summarize final N rollouts (default: 50)
      --output    tools/inferno_rl/tuning/tb_report.md  # omit to print to stdout
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Tags and their metadata
_TRAINING_HEALTH_TAGS = [
    "train/kl",
    "train/explained_variance",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/grad_norm",
    "train/loss",
]

_BEHAVIORAL_TAGS = [
    "rollout/deaths",
    "rollout/wave_timeouts",
    "rollout/waves_completed",
]

_PER_WAVE_TAGS = [
    "rollout/mean_wave_from_49",
    "rollout/mean_wave_from_55",
    "rollout/mean_wave_from_60",
    "rollout/mean_wave_from_63",
    "rollout/clear_rate_from_49",
    "rollout/clear_rate_from_55",
    "rollout/clear_rate_from_60",
    "rollout/clear_rate_from_63",
]

_CURRICULUM_TAGS = []

_ALL_TAGS = _TRAINING_HEALTH_TAGS + _BEHAVIORAL_TAGS + _PER_WAVE_TAGS + _CURRICULUM_TAGS


@dataclass
class TagSummary:
    tag: str
    total_events: int
    final_value: float
    mean_last_n: float
    mean_last_10: float
    trend: str  # "improving" | "declining" | "stable" | "insufficient_data"


def _detect_trend(mean_last_n: float, mean_last_10: float) -> str:
    """Trend is purely value-directional: 'improving' = value went up by >5%."""
    if abs(mean_last_n) < 1e-9:
        # Near-zero base: use absolute delta
        delta = mean_last_10 - mean_last_n
        if delta > 0.01:
            return "improving"
        if delta < -0.01:
            return "declining"
        return "stable"
    ratio = mean_last_10 / mean_last_n
    if ratio > 1.05:
        return "improving"
    if ratio < 0.95:
        return "declining"
    return "stable"


def _summarise_tag(values: list[float], last_n: int) -> tuple[float, float, float, str]:
    """Return (final_value, mean_last_n, mean_last_10, trend)."""
    taken = values[-last_n:] if len(values) >= last_n else values
    last_10 = values[-10:] if len(values) >= 10 else values

    final = values[-1]
    mean_n = float(np.mean(taken))
    mean_10 = float(np.mean(last_10))

    if len(values) < 10:
        trend = "insufficient_data"
    else:
        trend = _detect_trend(mean_n, mean_10)

    return final, mean_n, mean_10, trend


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


def _load_scalars(run_dir: Path) -> dict[str, list[float]]:
    """Load scalar values from TensorBoard event files. Returns tag → sorted list of values."""
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

    result: dict[str, list[float]] = {}
    available = set(ea.Tags().get("scalars", []))

    for tag in _ALL_TAGS:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        # Sort by step to ensure chronological order
        events_sorted = sorted(events, key=lambda e: e.step)
        result[tag] = [e.value for e in events_sorted]

    return result


def _fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _build_report(
    run_name: str,
    scalars: dict[str, list[float]],
    last_n: int,
) -> str:
    lines: list[str] = []

    total_events = max((len(v) for v in scalars.values()), default=0)
    lines.append(f"## TensorBoard Summary: {run_name}")
    lines.append(f"Last {last_n} rollouts of {total_events} total\n")

    def _table_for_tags(tags: list[str], show_last10: bool = True) -> list[str]:
        tbl = []
        if show_last10:
            tbl.append(f"| Metric | Final | Mean (last {last_n}) | Trend |")
            tbl.append("|--------|-------|---------------------|-------|")
        else:
            tbl.append(f"| Metric | Final | Mean (last {last_n}) |")
            tbl.append("|--------|-------|---------------------|")

        for tag in tags:
            if tag not in scalars:
                continue
            vals = scalars[tag]
            final, mean_n, mean_10, trend = _summarise_tag(vals, last_n)
            if show_last10:
                tbl.append(f"| {tag} | {_fmt(final)} | {_fmt(mean_n)} | {trend} |")
            else:
                tbl.append(f"| {tag} | {_fmt(final)} | {_fmt(mean_n)} |")
        return tbl

    # Training health
    lines.append("### Training Health")
    health_rows = _table_for_tags(_TRAINING_HEALTH_TAGS)
    if len(health_rows) > 2:
        lines.extend(health_rows)
    else:
        lines.append("_(no training health tags recorded)_")
    lines.append("")

    # Behavioral outcomes
    lines.append("### Behavioral Outcomes (per rollout)")
    behav_rows = _table_for_tags(_BEHAVIORAL_TAGS)
    if len(behav_rows) > 2:
        lines.extend(behav_rows)
    else:
        lines.append("_(no behavioral tags recorded)_")
    lines.append("")

    # Per-wave capability
    lines.append("### Per-Wave Capability (max wave reached this rollout)")
    wave_rows = _table_for_tags(_PER_WAVE_TAGS, show_last10=False)
    if len(wave_rows) > 2:
        lines.extend(wave_rows)
    else:
        lines.append("_(no per-wave tags recorded)_")
    lines.append("")

    # Curriculum
    lines.append("### Curriculum")
    curr_rows = _table_for_tags(_CURRICULUM_TAGS)
    if len(curr_rows) > 2:
        lines.extend(curr_rows)
    else:
        lines.append("_(no curriculum tags recorded — curriculum may be disabled)_")
    lines.append("")

    # Red flags
    lines.append("### Red Flags")
    flags: list[str] = []

    kl_vals = scalars.get("train/kl", [])
    if len(kl_vals) >= 10:
        kl_last10 = float(np.mean(kl_vals[-10:]))
        if kl_last10 > 0.05:
            flags.append(
                f"- `train/kl` last-10 mean = {kl_last10:.4f} > 0.05 — "
                "**KL divergence too high — consider halving lr**"
            )

    ev_vals = scalars.get("train/explained_variance", [])
    if len(ev_vals) >= 10:
        ev_last10 = float(np.mean(ev_vals[-10:]))
        if ev_last10 < 0.1:
            flags.append(
                f"- `train/explained_variance` last-10 mean = {ev_last10:.4f} < 0.1 — "
                "**Value function not learning — increase vf_coef**"
            )

    death_vals = scalars.get("rollout/deaths", [])
    if len(death_vals) >= 10:
        _, mean_n, _, trend = _summarise_tag(death_vals, last_n)
        if trend == "improving":
            flags.append("- `rollout/deaths` trend = improving (values increasing) — **Deaths increasing**")

    if flags:
        lines.extend(flags)
    else:
        lines.append("- None detected.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise TensorBoard training run to Markdown")
    parser.add_argument("--log-dir", required=True, help="Directory containing run subdirs")
    parser.add_argument("--run", default=None, help="Specific run subdir name; uses latest if omitted")
    parser.add_argument("--last-n", type=int, default=50, help="Summarise final N rollouts (default: 50)")
    parser.add_argument("--output", default=None, help="Output .md path; omit to print to stdout")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"ERROR: log-dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    run_dir, run_name = _load_run_dir(log_dir, args.run)
    print(f"Loading run: {run_name} ({run_dir})", file=sys.stderr)

    scalars = _load_scalars(run_dir)
    print(f"Loaded {len(scalars)} tags", file=sys.stderr)

    report = _build_report(run_name, scalars, args.last_n)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"Written to {out_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
