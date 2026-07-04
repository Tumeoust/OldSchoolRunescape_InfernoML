#!/usr/bin/env python3
"""Extract phase_failure_rate for specific step boundaries."""

import subprocess
import sys

# Checkpoints from V49_TB_TRACKING.md
checkpoints = [
    (0, "0.5M"),
    (500000, "2.9M"),
    (2900000, "8.4M"),
    (8400000, "13.7M"),
    (13700000, "19.1M"),
    (19100000, "24.5M"),
    (24500000, "29.9M"),
    (29900000, "35.3M"),
    (35300000, "40.6M"),
    (40600000, "46.0M"),
    (46000000, "51.4M"),
    (51400000, "56.8M"),
    (56800000, "62.2M"),
    (62200000, "67.6M"),
    (67600000, "72.9M"),
    (72900000, "78.3M"),
    (78300000, "83.7M"),
]

print("Step | Steps Label | phase_failure_rate")
print("-----|-------------|--------------------")

for prev_step, label in checkpoints:
    try:
        result = subprocess.run(
            ["python", "tools/inferno_rl/scripts/read_tb_metrics.py", str(prev_step)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        for line in result.stdout.split('\n'):
            if 'phase_failure_rate' in line:
                parts = line.split()
                avg_idx = parts.index('avg=') if 'avg=' in parts else -1
                if avg_idx >= 0:
                    avg_val = float(parts[avg_idx].replace('avg=', ''))
                    pct = avg_val * 100
                    print(f"{prev_step:>7} | {label:>11} | {pct:>6.1f}%")
                break
    except Exception as e:
        print(f"Error for {label}: {e}", file=sys.stderr)
