"""Train, evaluate against baselines, diagnose failures and write a report for one experiment.

Examples:
  python run_experiment.py --list-params
  python run_experiment.py --name stronger_crash --set reward.collision_penalty=30 --hypothesis "less obstacle hits"
  python run_experiment.py --name jitter --set env.start_jitter=0.3 reward.obstacle_margin=0.8 reward.obstacle_weight=1
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config as cfgmod
from diagnostics import (
    diagnose, greedy_policy, plot_reward_breakdown, plot_trajectories, plot_training,
    ppo_policy, random_policy, run_episodes, save_gif, strip_traj, summarize, write_report,
)
from training import train


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def make_run_dir(name):
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    run_dir = Path("runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", help="short experiment name (run.name)")
    p.add_argument("--hypothesis", help="what you expect this run to show (run.hypothesis)")
    p.add_argument("--timesteps", type=int, help="PPO timesteps (run.timesteps)")
    p.add_argument("--episodes", type=int, help="final evaluation episodes (run.eval_episodes)")
    p.add_argument("--seed", type=int, help="training seed (run.seed)")
    p.add_argument("--config", help="JSON file with a (partial) config; applied before --set")
    p.add_argument("--set", nargs="+", action="extend", default=[], metavar="SECTION.NAME=VALUE",
                   help="override any parameter, e.g. reward.collision_penalty=20")
    p.add_argument("--no-gif", action="store_true", help="skip GIF generation")
    p.add_argument("--list-params", action="store_true", help="print every tunable parameter and exit")
    return p.parse_args()


def build_config(args):
    cfg = cfgmod.Config()
    if args.config:
        file_cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        cfg = cfgmod.from_dict(file_cfg)
    shortcuts = {"name": args.name, "hypothesis": args.hypothesis, "timesteps": args.timesteps,
                 "eval_episodes": args.episodes, "seed": args.seed}
    overrides = [f"run.{k}={json.dumps(v)}" for k, v in shortcuts.items() if v is not None]
    cfgmod.apply_overrides(cfg, overrides + args.set)
    if args.no_gif:
        cfg.run.gif = False
    cfgmod.validate(cfg)
    return cfg


def main():
    args = parse_args()
    if args.list_params:
        cfgmod.print_params()
        return
    cfg = build_config(args)
    overrides = cfgmod.diff_from_default(cfg)

    run_dir = make_run_dir(cfg.run.name)
    commit = git_commit()
    save_json(run_dir / "config.json", cfg.to_dict())
    save_json(run_dir / "metadata.json", {
        "status": "running", "started_at": datetime.now().astimezone().isoformat(),
        "git_commit": commit, "python": sys.version, "command": sys.argv,
    })
    for source in ("environment.py", "config.py"):
        shutil.copy2(source, run_dir / source)

    print(f"\nExperiment: {cfg.run.name}  ->  {run_dir.resolve()}")
    if cfg.run.hypothesis:
        print(f"Hypothesis: {cfg.run.hypothesis}")
    changed = {k: v for k, v in overrides.items() if not k.startswith("run.")}
    print("Changes vs defaults:", json.dumps(changed) if changed else "none")

    try:
        print("\n1/4 Training PPO")
        model, monitor = train(cfg, run_dir)
        save_csv(run_dir / "training_episodes.csv", monitor.episodes)
        save_csv(run_dir / "eval_curve.csv", monitor.eval_curve)

        print("\n2/4 Evaluating PPO and baselines on identical seeds")
        run = cfg.run
        seed0 = run.seed + 1000
        policies = {
            "Random": random_policy,
            "Greedy": greedy_policy,
            "PPO": ppo_policy(model),
        }
        results = {name: run_episodes(cfg, factory, run.eval_episodes, seed0) for name, factory in policies.items()}
        summaries = {name: summarize(rows) for name, rows in results.items()}
        save_csv(run_dir / "episode_metrics_ppo.csv", strip_traj(results["PPO"]))
        save_csv(run_dir / "episode_metrics_baselines.csv",
                 [{"policy": n, **r} for n in ("Random", "Greedy") for r in strip_traj(results[n])])

        print("\n3/4 Diagnosing")
        diagnosis = diagnose(cfg, summaries, monitor.episodes, monitor.eval_curve)
        plot_trajectories(results, cfg, run_dir / "trajectories.png")
        plot_reward_breakdown(summaries["PPO"], run_dir / "reward_breakdown.png")
        plot_training(monitor.episodes, monitor.eval_curve, run.timesteps, run_dir / "training_curves.png")

        files = ["report.md", "results.json", "config.json", "trajectories.png", "reward_breakdown.png",
                 "training_curves.png", "training_episodes.csv", "eval_curve.csv", "episode_metrics_ppo.csv",
                 "episode_metrics_baselines.csv", "model.zip", "sb3_log/"]
        if run.gif:
            print("\n4/4 Recording GIFs")
            ppo_rows = results["PPO"]
            save_gif(cfg, ppo_rows[0], run_dir / "episode_first.gif", f"PPO seed {ppo_rows[0]['seed']}")
            files.append("episode_first.gif")
            failure = next((r for r in ppo_rows if not r["success"]), None)
            if failure is not None and failure is not ppo_rows[0]:
                save_gif(cfg, failure, run_dir / "episode_failure.gif", f"PPO seed {failure['seed']}")
                files.append("episode_failure.gif")

        save_json(run_dir / "results.json", {
            "name": run.name, "hypothesis": run.hypothesis, "seed": run.seed, "overrides": overrides,
            "summaries": summaries, "diagnosis": diagnosis, "eval_curve": monitor.eval_curve,
        })
        write_report(run_dir / "report.md", cfg, overrides, summaries, diagnosis, monitor.episodes, files)
        save_json(run_dir / "metadata.json", {
            "status": "completed", "completed_at": datetime.now().astimezone().isoformat(), "git_commit": commit,
        })
    except Exception as error:
        save_json(run_dir / "metadata.json", {
            "status": "failed", "failed_at": datetime.now().astimezone().isoformat(), "error": repr(error),
            "git_commit": commit,
        })
        raise

    print("\n" + "=" * 70)
    for name, s in summaries.items():
        lo, hi = s["success_ci95"]
        print(f"{name:<7} success {s['success_rate']:.0%} [{lo:.0%}-{hi:.0%}]  collision {s['collision_rate']:.0%}  "
              f"timeout {s['outcome_rates']['timeout']:.0%}  return {s['mean_reward']:.1f}")
    print("\nDiagnosis:")
    for d in diagnosis:
        print(f"  - {d['finding']}\n      Try: {d['suggestion']}")
    print(f"\nFull report: {(run_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
