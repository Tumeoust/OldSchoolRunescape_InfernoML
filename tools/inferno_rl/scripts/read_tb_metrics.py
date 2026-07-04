"""Read TensorBoard event file and print averaged metrics since a given step."""

import math
import os
import struct
import sys
from collections import defaultdict

from tensorboard.compat.proto.event_pb2 import Event

TAGS_ORDERED = [
    "rollout/num_episodes",
    "rollout/deaths",
    "rollout/wave_timeouts",
    "rollout/phase_failure_rate",
    "raw_reward_terms/clear_rate_from_1",
    "train/explained_variance",
    "train/kl",
    "train/value_loss",
    "train/clip_fraction",
    "train/entropy_coef",
    "rollout/return_mean",
    "rollout/running_reward_var",
    "train/grad_norm",
    "train/early_stop",
]
TAGS_OF_INTEREST = set(TAGS_ORDERED)

REWARD_TERM_PREFIX = "raw_reward_terms/ep_sum_mean/"


def find_event_file():
    log_dirs = sorted(
        [d for d in os.listdir("logs") if d.startswith("V")],
        key=lambda x: int(x.split("_")[0][1:]),
    )
    log_dir = os.path.join("logs", log_dirs[-1])

    # Find all event files and return the newest by modification time
    event_files = []
    for root, _, files in os.walk(log_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                event_files.append(os.path.join(root, f))

    if event_files:
        # Return the event file with the most recent modification time
        return max(event_files, key=lambda x: os.path.getmtime(x))
    return None


def scan_events(event_file, prev_step):
    filesize = os.path.getsize(event_file)
    print(f"Event file: {event_file}")
    print(f"Size: {filesize / 1024 / 1024:.1f} MB")
    print(f"Averaging since step > {prev_step}")

    values = defaultdict(list)
    max_step = 0
    chunk_size = 50 * 1024 * 1024
    overlap = 1 * 1024 * 1024

    with open(event_file, "rb") as fh:
        offset = 0
        while offset < filesize:
            fh.seek(offset)
            data = fh.read(chunk_size + overlap)
            pos = 0
            while pos < min(chunk_size, len(data)) - 12:
                try:
                    length = struct.unpack("Q", data[pos : pos + 8])[0]
                    if length > 1000000 or length == 0:
                        pos += 1
                        continue
                    start = pos + 12
                    end = start + length
                    if end + 4 > len(data):
                        break
                    event = Event()
                    event.ParseFromString(data[start:end])
                    if event.step > max_step:
                        max_step = event.step
                    if event.HasField("summary") and event.step > prev_step:
                        for v in event.summary.value:
                            if v.HasField("simple_value") and math.isfinite(v.simple_value):
                                if v.tag in TAGS_OF_INTEREST or v.tag.startswith(REWARD_TERM_PREFIX):
                                    values[v.tag].append(v.simple_value)
                    pos = end + 4
                except Exception:
                    pos += 1
            offset += chunk_size

    return values, max_step


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <prev_step>")
        sys.exit(1)

    prev_step = int(sys.argv[1])
    event_file = find_event_file()
    if not event_file:
        print("ERROR: No event file found")
        sys.exit(1)

    values, max_step = scan_events(event_file, prev_step)

    print(f"Latest step: {max_step}")
    print()
    for tag in TAGS_ORDERED:
        vals = values.get(tag, [])
        if vals:
            avg = sum(vals) / len(vals)
            print(f"{tag:<45} n={len(vals):>5}  avg={avg:>12.4f}")
        else:
            print(f"{tag:<45} NO DATA in range")

    print()
    print("--- Reward Terms (ep_sum_mean) ---")
    reward_tags = sorted(k for k in values if k.startswith(REWARD_TERM_PREFIX))
    for tag in reward_tags:
        vals = values[tag]
        avg = sum(vals) / len(vals)
        short = tag[len(REWARD_TERM_PREFIX):]
        print(f"{short:<35} n={len(vals):>5}  avg={avg:>12.4f}")
    if not reward_tags:
        print("(no reward term data in range)")


if __name__ == "__main__":
    main()
