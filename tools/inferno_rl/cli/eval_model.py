"""
JSON-output model evaluation wrapper.

Usage:
    python -m tools.inferno_rl.cli.eval_model \
      --model models/V21_climb/inferno_gpu_w55-66_..._6103.pt \
      --episodes 100 --start-wave 49 --max-wave 66 --seed 0 \
      --output-format json
"""

import argparse
import json
import sys

from ..eval import load_model
from ..death_analysis import run_death_analysis, print_histogram


def main() -> None:
    parser = argparse.ArgumentParser(description="Eval Inferno RL model with JSON output")
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-wave", type=int, default=49)
    parser.add_argument("--max-wave", type=int, default=66)
    parser.add_argument("--seed", type=int, default=0, help="Base seed offset")
    parser.add_argument("--output-format", choices=["json", "text"], default="text",
                        help="Output format (json or text)")
    args = parser.parse_args()

    model = load_model(args.model)
    death_waves = run_death_analysis(
        model, args.start_wave, args.max_wave, args.episodes, args.seed
    )

    if args.output_format == "text":
        print_histogram(args.model, args.start_wave, args.max_wave,
                        args.episodes, args.seed, death_waves)
        return

    # Build JSON output
    cleared = death_waves[0]
    total_died = sum(v for k, v in death_waves.items() if k > 0)
    total_timeout = sum(v for k, v in death_waves.items() if k < 0)
    n = args.episodes

    per_wave = {}
    for wave in range(args.start_wave, args.max_wave + 1):
        deaths = death_waves.get(wave, 0)
        timeouts = death_waves.get(-wave, 0)
        if deaths > 0 or timeouts > 0:
            per_wave[str(wave)] = {"deaths": deaths, "timeouts": timeouts}

    result = {
        "model": args.model,
        "episodes": n,
        "start_wave": args.start_wave,
        "max_wave": args.max_wave,
        "seed": args.seed,
        "clear_rate": round(cleared / n, 4),
        "death_rate": round(total_died / n, 4),
        "timeout_rate": round(total_timeout / n, 4),
        "cleared": cleared,
        "died": total_died,
        "timed_out": total_timeout,
        "per_wave": per_wave,
    }

    json.dump(result, sys.stdout, indent=2)
    print()  # trailing newline


if __name__ == "__main__":
    main()
