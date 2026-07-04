"""
Benchmark simulator performance (steps/sec).

Measures the raw simulator throughput by running N steps with random actions.
Run before and after optimizations to measure speedup.

Usage:
    python -m tools.inferno_rl.benchmark_sim         # Default: 10k steps
    python -m tools.inferno_rl.benchmark_sim 50000   # Custom step count
"""
import sys
import time
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.inferno_rl.simulator.simulator import InfernoSimulator


def benchmark(n_steps: int = 10000, start_wave: int = 55, max_wave: int = 64):
    sim = InfernoSimulator(start_wave=start_wave, max_wave=max_wave)
    sim.reset()

    resets = 0
    start = time.perf_counter()

    for _ in range(n_steps):
        action = random.randint(0, 42)
        result = sim.step(action)
        if result.is_terminal():
            sim.reset()
            resets += 1

    elapsed = time.perf_counter() - start
    steps_per_sec = n_steps / elapsed

    # Check which geometry module is loaded on the same import path training uses.
    from tools.inferno_rl.simulator import geometry as geo_mod
    is_cython = hasattr(geo_mod, "__pyx_capi__") or ".pyd" in getattr(geo_mod, "__file__", "")
    backend = "Cython (.pyd)" if is_cython else "Pure Python (.py)"

    print(f"Backend:    {backend}")
    print(f"Steps:      {n_steps:,}")
    print(f"Resets:     {resets}")
    print(f"Time:       {elapsed:.2f}s")
    print(f"Speed:      {steps_per_sec:,.0f} steps/sec")

    return steps_per_sec


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    benchmark(n_steps=n)
