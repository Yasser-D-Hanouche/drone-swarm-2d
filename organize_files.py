#!/usr/bin/env python3
"""
Safely organize common output files in the drone-swarm project.

Default mode is a dry run: it prints planned moves without changing files.
Run with --apply to perform the moves.

By default, it only scans the project root (not subdirectories), so existing
runs/, .venv/, .git/, and other folders are left untouched.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CATEGORIES = {
    "models": {".zip", ".pt", ".pth", ".ckpt", ".onnx"},
    "animations": {".gif", ".mp4", ".webm", ".avi"},
    "plots": {".png", ".jpg", ".jpeg", ".svg", ".pdf"},
    "logs": {".log", ".out", ".err"},
    "data": {".csv", ".jsonl", ".parquet"},
}

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "runs",
    "models", "animations", "plots", "logs", "data", "scripts",
}

SKIP_NAMES = {
    "environment.py",
    "train.py",
    "run_experiment.py",
    "animate.py",
    "animate_trained.py",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    ".gitignore",
}


def unique_destination(destination: Path) -> Path:
    """Avoid overwriting an existing file by adding a numeric suffix."""
    if not destination.exists():
        return destination

    counter = 1
    while True:
        candidate = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Organize output files in the project root."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project directory (defaults to the current directory).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually move files. Without this flag, only preview changes.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    print(f"Project directory: {root}")
    print("Mode:", "APPLY CHANGES" if args.apply else "DRY RUN (no files moved)")
    print()

    planned: list[tuple[Path, Path]] = []

    for item in sorted(root.iterdir()):
        if item.is_dir() or item.name in SKIP_NAMES or item.name.startswith("."):
            continue

        category = next(
            (name for name, extensions in CATEGORIES.items()
             if item.suffix.lower() in extensions),
            None,
        )
        if category is None:
            continue

        destination_dir = root / category
        destination = unique_destination(destination_dir / item.name)
        planned.append((item, destination))

    if not planned:
        print("No matching files found in the project root.")
        print("Note: this script intentionally does not scan inside existing folders.")
        return 0

    for source, destination in planned:
        print(f"{source.name}  ->  {destination.relative_to(root)}")

    if not args.apply:
        print(f"\nPreview only: {len(planned)} file(s) would be moved.")
        print("To perform these moves, rerun with --apply.")
        return 0

    moved = 0
    for source, destination in planned:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Re-check at move time in case a destination appeared after planning.
        destination = unique_destination(destination)
        shutil.move(str(source), str(destination))
        moved += 1

    print(f"\nMoved {moved} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())