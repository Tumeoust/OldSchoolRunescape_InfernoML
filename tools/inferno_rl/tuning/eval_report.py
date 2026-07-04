"""
Runs eval against a checkpoint, compares to baseline.json, and writes structured results.

Usage:
    python -m tools.inferno_rl.tuning.eval_report
      --model     tools/inferno_rl/models/smoke/<name>.pt
      --baseline  tools/inferno_rl/tuning/baseline.json   # default
      --episodes  200
      --start-waves 35,46,55,63
      --output-md   tools/inferno_rl/tuning/eval_report.md
      --output-json tools/inferno_rl/tuning/result_N.json
      --label     "Attempt N: description"

    # Populate real baseline numbers from V10 checkpoint:
    python -m tools.inferno_rl.tuning.eval_report
      --model     tools/inferno_rl/models/V10/inferno_gpu_w35-66_20260221_094121_122.pt
      --update-baseline
      --episodes  200
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_DEFAULT_BASELINE = Path(__file__).parent / "baseline.json"
_DEFAULT_START_WAVES = [35, 46, 55, 63]
_DEFAULT_MAX_WAVE = 66


def _compute_metrics(stats: dict, n: int) -> dict:
    return {
        "death_rate": round(stats["deaths"] / n, 4),
        "cleared_start_rate": round(stats["wave_cleared_start"] / n, 4),
        "complete_rate": round(stats["completions"] / n, 4),
        "mean_wave": round(float(np.mean(stats["max_waves"])), 2),
        "median_wave": int(np.median(stats["max_waves"])),
    }


def _determine_verdict(per_wave: dict) -> str:
    """
    IMPROVEMENT: ≥1 wave has death_rate delta ≤ -0.02 AND no wave has delta > +0.05
    REGRESSION:  any wave has death_rate delta > +0.05
    NEUTRAL:     neither
    """
    has_improvement = False
    for wave_data in per_wave.values():
        delta = wave_data["delta"]["death_rate"]
        if delta > 0.05:
            return "REGRESSION"
        if delta <= -0.02:
            has_improvement = True
    return "IMPROVEMENT" if has_improvement else "NEUTRAL"


def _fmt_delta(v: float, is_rate: bool = False) -> str:
    sign = "+" if v >= 0 else ""
    if is_rate:
        return f"{sign}{v * 100:.1f}pp"
    return f"{sign}{v:.2f}"


def _build_md_report(
    label: str,
    model_path: str,
    baseline: dict,
    per_wave: dict,
    verdict: str,
    episodes: int,
) -> str:
    lines: list[str] = []
    lines.append(f"## Eval Report: {label}")
    lines.append(f"**Model**: `{model_path}`  ")
    lines.append(f"**Baseline**: {baseline['version']} ({baseline['checkpoint']})  ")
    lines.append(f"**Episodes**: {episodes} per start wave  ")
    lines.append(f"**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    lines.append(f"**Verdict**: {verdict}")
    lines.append("")

    lines.append("### Per-Wave Results vs Baseline")
    lines.append("")
    lines.append("| Start Wave | Metric | Baseline | Current | Delta |")
    lines.append("|------------|--------|----------|---------|-------|")

    for wave_str, data in sorted(per_wave.items(), key=lambda x: int(x[0])):
        b = data["baseline"]
        c = data["current"]
        d = data["delta"]
        lines.append(
            f"| W{wave_str} | death_rate | {b['death_rate']*100:.1f}% | "
            f"{c['death_rate']*100:.1f}% | {_fmt_delta(d['death_rate'], True)} |"
        )
        lines.append(
            f"| W{wave_str} | cleared_start | {b['cleared_start_rate']*100:.1f}% | "
            f"{c['cleared_start_rate']*100:.1f}% | {_fmt_delta(d['cleared_start_rate'], True)} |"
        )
        lines.append(
            f"| W{wave_str} | mean_wave | {b['mean_wave']:.1f} | "
            f"{c['mean_wave']:.1f} | {_fmt_delta(d['mean_wave'])} |"
        )

    lines.append("")

    lines.append("### Raw Current Metrics")
    lines.append("| Start Wave | Deaths | Cleared Start | Complete | Mean Wave | Median |")
    lines.append("|------------|--------|---------------|----------|-----------|--------|")
    for wave_str, data in sorted(per_wave.items(), key=lambda x: int(x[0])):
        c = data["current"]
        lines.append(
            f"| W{wave_str} | {c['death_rate']*100:.1f}% | "
            f"{c['cleared_start_rate']*100:.1f}% | "
            f"{c['complete_rate']*100:.1f}% | "
            f"{c['mean_wave']:.1f} | {c['median_wave']} |"
        )

    return "\n".join(lines) + "\n"


def run_eval_report(
    model_path: str,
    baseline_path: Path,
    episodes: int,
    start_waves: list[int],
    max_wave: int,
    label: str,
    output_md: Path | None,
    output_json: Path | None,
    update_baseline: bool,
) -> None:
    from ..eval import eval_model

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    print(f"\nRunning eval: {label}")
    print(f"Model: {model_path}")
    print(f"Episodes: {episodes} per wave, waves: {start_waves}")

    raw_results = eval_model(label, model_path, start_waves, max_wave, episodes)

    if update_baseline:
        new_results: dict[str, dict] = {}
        for sw in start_waves:
            stats = raw_results[sw]
            new_results[str(sw)] = _compute_metrics(stats, episodes)
        baseline["results"] = new_results
        baseline["checkpoint"] = model_path
        baseline["episodes"] = episodes
        baseline["date"] = datetime.now().strftime("%Y-%m-%d")
        baseline.pop("notes", None)
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        print(f"\nBaseline updated: {baseline_path}")
        return

    per_wave: dict[str, dict] = {}
    for sw in start_waves:
        stats = raw_results[sw]
        current = _compute_metrics(stats, episodes)
        sw_str = str(sw)
        b_metrics = baseline["results"].get(sw_str, {})
        delta = {
            k: round(current[k] - b_metrics.get(k, 0.0), 4)
            for k in ("death_rate", "cleared_start_rate", "complete_rate", "mean_wave")
        }
        per_wave[sw_str] = {
            "baseline": b_metrics,
            "current": current,
            "delta": delta,
        }

    verdict = _determine_verdict(per_wave)

    md = _build_md_report(label, model_path, baseline, per_wave, verdict, episodes)

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md, encoding="utf-8")
        print(f"\nMarkdown report: {output_md}")
    else:
        print(md)

    result_json = {
        "label": label,
        "checkpoint": model_path,
        "baseline_version": baseline["version"],
        "episodes": episodes,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "per_wave": per_wave,
        "verdict": verdict,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
        print(f"JSON result:     {output_json}")
    else:
        print(json.dumps(result_json, indent=2))

    print(f"\nVerdict: {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval, compare to baseline, write report")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--baseline",
        default=str(_DEFAULT_BASELINE),
        help="Path to baseline.json (default: tuning/baseline.json)",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument(
        "--start-waves",
        default=None,
        help="Comma-separated start waves (default: 35,46,55,63)",
    )
    parser.add_argument("--max-wave", type=int, default=_DEFAULT_MAX_WAVE)
    parser.add_argument("--label", default="Unlabelled eval", help="Human-readable label for this run")
    parser.add_argument("--output-md", default=None, help="Output Markdown report path")
    parser.add_argument("--output-json", default=None, help="Output JSON result path")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Run eval and overwrite baseline.json with real numbers instead of comparing",
    )
    args = parser.parse_args()

    start_waves = (
        [int(w) for w in args.start_waves.split(",")]
        if args.start_waves
        else _DEFAULT_START_WAVES
    )

    run_eval_report(
        model_path=args.model,
        baseline_path=Path(args.baseline),
        episodes=args.episodes,
        start_waves=start_waves,
        max_wave=args.max_wave,
        label=args.label,
        output_md=Path(args.output_md) if args.output_md else None,
        output_json=Path(args.output_json) if args.output_json else None,
        update_baseline=args.update_baseline,
    )


if __name__ == "__main__":
    main()
