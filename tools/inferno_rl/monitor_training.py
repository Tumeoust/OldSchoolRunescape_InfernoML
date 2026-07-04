"""
V20 training monitor. Checks TB metrics every 5 minutes, detects stalls, logs snapshots.
Run from repo root: python -m tools.inferno_rl.monitor_training
"""

import glob
import os
import sys
import time
from datetime import datetime

CHECK_INTERVAL = 300  # 5 minutes
TARGET_STEPS = 80_000_000
LOGDIR_PATTERN = "logs/V20_climb/*"
TRACKING_DOC = "tools/inferno_rl/docs/V20_TB_TRACKING.md"

# Milestone thresholds (steps) — write a full snapshot row at each
MILESTONES = [5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000,
              50_000_000, 60_000_000, 70_000_000, 80_000_000]

# Stall detection thresholds
MIN_EV = 0.60                    # EV below this for 2M+ steps = value divergence
MAX_STEPS_NO_FRONTIER = 10_000_000  # Frontier stuck for 10M steps = stalled
DEATH_REGRESSION_WINDOW = 5_000_000  # Deaths increasing over 5M steps = bad
ENTROPY_COLLAPSE_THRESHOLD = -0.02   # Entropy loss near 0 = collapsed


def get_metrics(logdir: str) -> dict | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(logdir)
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return None

    def last(tag):
        events = ea.Scalars(tag) if tag in tags else []
        return (events[-1].value, events[-1].step) if events else (None, None)

    def last_n(tag, n=10):
        events = ea.Scalars(tag) if tag in tags else []
        return [(e.value, e.step) for e in events[-n:]]

    m = {}
    m["steps"] = last("train/total_steps")[0]
    m["deaths_val"], m["deaths_step"] = last("rollout/deaths")
    m["waves_val"], _ = last("rollout/waves_completed")
    m["timeouts_val"], _ = last("rollout/wave_timeouts")
    m["frontier_val"], m["frontier_step"] = last("rollout/curriculum_frontier_max")
    m["frontier_mean"], _ = last("rollout/curriculum_frontier_mean")
    m["mean_reward"], _ = last("rollout/reward/mean_episode_reward")
    m["max_reward"], _ = last("rollout/reward/max_episode_reward")
    m["mean_ep_len"], _ = last("rollout/len/mean_episode_length")
    m["ev"], _ = last("train/explained_variance")
    m["entropy"], _ = last("train/entropy_loss")
    m["grad_norm"], _ = last("train/grad_norm")
    m["kl"], _ = last("train/kl")
    m["entropy_coef"], _ = last("train/entropy_coef")
    m["fps"], _ = last("rollout/fps")

    # History for trend detection
    m["deaths_history"] = last_n("rollout/deaths", 20)
    m["ev_history"] = last_n("train/explained_variance", 20)
    m["frontier_history"] = last_n("rollout/curriculum_frontier_max", 50)
    m["entropy_history"] = last_n("train/entropy_loss", 20)
    m["reward_history"] = last_n("rollout/reward/mean_episode_reward", 20)

    return m


def check_stall(m: dict) -> str | None:
    """Return a stall reason string, or None if training looks healthy."""
    steps = m["steps"] or 0

    # 1. Entropy collapse
    if m["entropy"] is not None and m["entropy"] > ENTROPY_COLLAPSE_THRESHOLD:
        return f"ENTROPY COLLAPSE: entropy_loss={m['entropy']:.4f} (near 0 = deterministic policy)"

    # 2. EV divergence — check if EV has been below threshold for recent history
    if m["ev_history"]:
        recent_ev = [v for v, s in m["ev_history"]]
        if len(recent_ev) >= 5 and all(v < MIN_EV for v in recent_ev[-5:]):
            return f"EV DIVERGENCE: last 5 readings all below {MIN_EV} ({recent_ev[-5:]})"

    # 3. Frontier stuck
    if m["frontier_history"] and steps > MIN_STEPS_FOR_FRONTIER_CHECK:
        first_frontier = m["frontier_history"][0]
        last_frontier = m["frontier_history"][-1]
        step_span = last_frontier[1] - first_frontier[1]
        if step_span >= MAX_STEPS_NO_FRONTIER and first_frontier[0] == last_frontier[0]:
            return (f"FRONTIER STALLED: stuck at W{int(last_frontier[0])} for "
                    f"{step_span/1e6:.1f}M steps")

    # 4. Deaths increasing monotonically over window
    if m["deaths_history"]:
        recent = [v for v, s in m["deaths_history"]]
        if len(recent) >= 10:
            first_half = sum(recent[:5]) / 5
            second_half = sum(recent[-5:]) / 5
            step_span = m["deaths_history"][-1][1] - m["deaths_history"][0][1]
            if step_span >= DEATH_REGRESSION_WINDOW and second_half > first_half * 1.5:
                return (f"DEATH REGRESSION: deaths trending up "
                        f"({first_half:.0f} -> {second_half:.0f} over {step_span/1e6:.1f}M steps)")

    return None


# Need enough data before checking frontier stall
MIN_STEPS_FOR_FRONTIER_CHECK = 5_000_000


def format_status(m: dict, check_num: int) -> str:
    steps = m["steps"] or 0
    eta_hr = (TARGET_STEPS - steps) / (m["fps"] or 1) / 3600 if m["fps"] else "?"
    now = datetime.now().strftime("%H:%M:%S")
    frontier_text = "n/a"
    if m["frontier_val"] is not None:
        frontier_text = f"{m['frontier_val']:.0f}"
        if m["frontier_mean"] is not None:
            frontier_text += f" (mean {m['frontier_mean']:.2f})"

    lines = [
        f"=== Check #{check_num} @ {now} | {steps/1e6:.1f}M / {TARGET_STEPS/1e6:.0f}M steps | ETA: {eta_hr:.1f}h ===",
        f"  frontier: {frontier_text} | "
        f"deaths: {m['deaths_val']:.0f} | waves: {m['waves_val']:.0f} | timeouts: {m['timeouts_val']:.0f}",
        f"  reward: mean={m['mean_reward']:.2f} max={m['max_reward']:.2f} | ep_len: {m['mean_ep_len']:.0f}",
        f"  EV: {m['ev']:.3f} | entropy: {m['entropy']:.4f} (coef={m['entropy_coef']:.4f}) | "
        f"grad: {m['grad_norm']:.2f} | KL: {m['kl']:.4f}",
        f"  fps: {m['fps']:.0f}",
    ]

    # Reward trend
    if m["reward_history"] and len(m["reward_history"]) >= 6:
        recent = [v for v, s in m["reward_history"]]
        early = sum(recent[:3]) / 3
        late = sum(recent[-3:]) / 3
        trend = "UP" if late > early + 0.3 else "DOWN" if late < early - 0.3 else "FLAT"
        lines.append(f"  reward trend: {trend} ({early:.2f} -> {late:.2f})")

    return "\n".join(lines)


def _milestone_label(steps: float) -> str:
    return f"{steps / 1e6:.0f}M"


def _reward_trend(m: dict) -> str:
    if m["reward_history"] and len(m["reward_history"]) >= 6:
        recent = [v for v, s in m["reward_history"]]
        early = sum(recent[:3]) / 3
        late = sum(recent[-3:]) / 3
        if late > early + 0.3:
            return "UP"
        elif late < early - 0.3:
            return "DOWN"
    return "FLAT"


def write_milestone(m: dict, label: str, note: str = "") -> None:
    """Append a milestone row to the tracking document."""
    steps = m["steps"] or 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    row = (
        f"| {label} | {now} | {m['frontier_val']:.0f} | {m['frontier_mean']:.2f} "
        f"| {m['deaths_val']:.0f} | {m['waves_val']:.0f} | {m['timeouts_val']:.0f} "
        f"| {m['mean_reward']:.2f} | {m['mean_ep_len']:.0f} "
        f"| {m['ev']:.3f} | {m['entropy']:.4f} | {m['grad_norm']:.2f} "
        f"| {_reward_trend(m)} | {note} |"
    )

    # Append to tracking doc
    if not os.path.exists(TRACKING_DOC):
        print(f"  WARNING: {TRACKING_DOC} not found, skipping file write")
        return

    with open(TRACKING_DOC, "r", encoding="utf-8") as f:
        content = f.read()

    # If the auto-log table doesn't exist yet, create it
    if "<!-- AUTO-LOG -->" not in content:
        table_header = (
            "\n---\n\n## Auto-Logged Milestones\n\n"
            "<!-- AUTO-LOG -->\n"
            "| Steps | Time | Frontier | F.Mean | Deaths | Waves | Timeouts "
            "| MeanRwd | EpLen | EV | Entropy | Grad | Trend | Note |\n"
            "|-------|------|----------|--------|--------|-------|----------"
            "|---------|-------|----|---------|------|----|------|\n"
            "<!-- END-AUTO-LOG -->\n"
        )
        content += table_header

    # Insert row before the END marker
    content = content.replace("<!-- END-AUTO-LOG -->", row + "\n<!-- END-AUTO-LOG -->")

    with open(TRACKING_DOC, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  -> Wrote milestone {label} to {TRACKING_DOC}")


def write_stall(m: dict, reason: str) -> None:
    """Append a STALL entry to the tracking document."""
    steps = m["steps"] or 0
    label = f"{steps / 1e6:.1f}M"
    write_milestone(m, label, note=f"**STALL: {reason}**")


def main():
    print(f"V20 Training Monitor started. Checking every {CHECK_INTERVAL}s until {TARGET_STEPS/1e6:.0f}M steps.")
    print(f"Stall detection: EV<{MIN_EV} for 5 readings, frontier stuck {MAX_STEPS_NO_FRONTIER/1e6:.0f}M steps, "
          f"entropy>{ENTROPY_COLLAPSE_THRESHOLD}")
    print(f"Writing milestones to: {TRACKING_DOC}")
    print()

    dirs = glob.glob(LOGDIR_PATTERN)
    if not dirs:
        print(f"ERROR: No log directory found matching {LOGDIR_PATTERN}")
        sys.exit(1)
    logdir = dirs[0]
    print(f"Monitoring: {logdir}")
    print()

    check_num = 0

    # Seed milestones_hit from current step count so restarts don't re-fire
    try:
        init_m = get_metrics(logdir)
        init_steps = init_m["steps"] if init_m and init_m["steps"] else 0
    except Exception:
        init_steps = 0
    milestones_hit: set[int] = {ms for ms in MILESTONES if ms <= init_steps}
    if milestones_hit:
        print(f"Already past milestones: {sorted(ms // 1_000_000 for ms in milestones_hit)}M")
        print()

    while True:
        check_num += 1
        try:
            m = get_metrics(logdir)
        except Exception as e:
            print(f"Check #{check_num}: Error reading metrics: {e}")
            time.sleep(CHECK_INTERVAL)
            continue

        if m is None or m["steps"] is None:
            print(f"Check #{check_num}: No data yet, waiting...")
            time.sleep(CHECK_INTERVAL)
            continue

        steps = m["steps"]
        print(format_status(m, check_num))

        # Write milestone rows at key step thresholds
        for ms in MILESTONES:
            if ms not in milestones_hit and steps >= ms:
                milestones_hit.add(ms)
                write_milestone(m, _milestone_label(ms))

        # Check for stalls
        stall = check_stall(m)
        if stall:
            print(f"\n  *** STALL DETECTED: {stall} ***")
            print(f"  Stopping monitor. Review TB logs and decide whether to continue or abort.")
            write_stall(m, stall)
            sys.exit(1)

        # Check if target reached
        if steps >= TARGET_STEPS:
            print(f"\n  TARGET REACHED: {steps/1e6:.1f}M steps. Training complete (or close).")
            write_milestone(m, _milestone_label(steps), note="TARGET REACHED")
            sys.exit(0)

        print()
        sys.stdout.flush()
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
