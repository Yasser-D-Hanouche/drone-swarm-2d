"""Compare experiments in runs/. Runs sharing a name are treated as seeds and shown as mean ± std.

  python compare_runs.py            # all runs
  python compare_runs.py --last 5   # most recent 5 experiment names
  python compare_runs.py --md       # also write runs/comparison.md
"""
import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


def load_runs(root):
    groups = OrderedDict()
    for path in sorted(root.glob("*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        groups.setdefault(data["name"], []).append(data)
    return groups


def cell(values, fmt):
    arr = np.asarray(values, dtype=float)
    return f"{fmt.format(arr.mean())}" + (f" ± {fmt.format(arr.std())}" if len(arr) > 1 else "")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--last", type=int, help="only the N most recent experiment names")
    parser.add_argument("--md", action="store_true", help="write runs/comparison.md")
    args = parser.parse_args()

    groups = load_runs(Path(args.runs_dir))
    if not groups:
        print("No completed runs found.")
        return
    names = list(groups)[-args.last:] if args.last else list(groups)

    header = ["experiment", "seeds", "changes", "success", "collision", "timeout", "return", "final dist"]
    rows = []
    for name in names:
        runs = groups[name]
        ppo = [r["summaries"]["PPO"] for r in runs]
        changes = {k: v for k, v in runs[0]["overrides"].items() if not k.startswith("run.")}
        rows.append([
            name, str(len(runs)), " ".join(f"{k}={v}" for k, v in changes.items()) or "-",
            cell([s["success_rate"] * 100 for s in ppo], "{:.0f}%"),
            cell([s["collision_rate"] * 100 for s in ppo], "{:.0f}%"),
            cell([s["outcome_rates"]["timeout"] * 100 for s in ppo], "{:.0f}%"),
            cell([s["mean_reward"] for s in ppo], "{:.1f}"),
            cell([s["mean_final_dist"] for s in ppo], "{:.2f}"),
        ])

    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths))
    print(line(header))
    print(line(["-" * w for w in widths]))
    for r in rows:
        print(line(r))
    if any(r[1] == "1" for r in rows):
        print("\nNote: single-seed rows are noisy; repeat with --seed to tell real effects from luck.")

    if args.md:
        out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        out += ["| " + " | ".join(r) + " |" for r in rows]
        Path(args.runs_dir, "comparison.md").write_text("\n".join(out) + "\n", encoding="utf-8")
        print(f"\nWrote {Path(args.runs_dir, 'comparison.md')}")


if __name__ == "__main__":
    main()
