"""Show how far a (running) training run is, from its sb3_log/progress.csv.

    python cluster/progress.py                     # newest folder in runs/
    python cluster/progress.py runs/<run_folder>   # a specific run
    watch -n 30 python cluster/progress.py         # refresh every 30 s (Ctrl+C to stop)

Uses only the standard library, so any python3 works (no venv needed).
"""
import csv
import json
import sys
from pathlib import Path

runs = Path(__file__).resolve().parent.parent / "runs"
run = Path(sys.argv[1]) if len(sys.argv) > 1 else max(runs.iterdir(), key=lambda p: p.name)
log = run / "sb3_log" / "progress.csv"
if not log.exists():
    sys.exit(f"{run.name}: no progress yet (training has not finished its first update)")

rows = list(csv.DictReader(open(log)))
if not rows:
    sys.exit(f"{run.name}: no progress yet")
cfg = json.load(open(run / "config.json"))
last = rows[-1]
total = cfg["run"]["timesteps"]
steps = int(float(last["time/total_timesteps"])) // cfg["env"]["num_drones"]  # SB3 counts one step per drone
elapsed = float(last["time/time_elapsed"])
eta = (total - steps) * elapsed / max(steps, 1)

evals = [r for r in rows if r.get("eval/success_rate")]
status = json.load(open(run / "metadata.json")).get("status", "?")

print(f"{run.name}  [{status}]")
print(f"  progress  {steps:,}/{total:,} env steps ({steps / total:.1%})")
print(f"  time      {elapsed / 60:.1f} min elapsed, ~{eta / 60:.1f} min left in training (evaluation and report follow)")
if last.get("outcome/success"):
    print(f"  training  success {float(last['outcome/success']):.0%} in the last rollout")
if evals:
    print(f"  eval      success {float(evals[-1]['eval/success_rate']):.0%} at the latest periodic evaluation")
