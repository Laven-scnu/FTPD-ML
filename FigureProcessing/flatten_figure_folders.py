"""Flatten nested extracted figure folders into a single directory."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


def collect_leaf_directories(source_root: Path) -> list[Path]:
    leaf_dirs: list[Path] = []
    for paper_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for figure_dir in sorted(path for path in paper_dir.iterdir() if path.is_dir()):
            leaf_dirs.append(figure_dir)
    return leaf_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten paper/figure directory layout into a single figure directory.")
    parser.add_argument("--source-dir", required=True, type=Path, help="Input directory containing paper subdirectories.")
    parser.add_argument("--target-dir", required=True, type=Path, help="Output directory for flattened figure folders.")
    parser.add_argument("--move", action="store_true", help="Move folders instead of copying them.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip target folders that already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {args.source_dir}")

    args.target_dir.mkdir(parents=True, exist_ok=True)
    figure_dirs = collect_leaf_directories(args.source_dir)

    copied = 0
    skipped = 0
    for figure_dir in tqdm(figure_dirs, desc="Flattening folders", unit="folder"):
        destination = args.target_dir / figure_dir.name
        if destination.exists():
            skipped += 1
            if args.skip_existing:
                continue
            raise FileExistsError(f"Target already exists: {destination}")

        if args.move:
            shutil.move(str(figure_dir), str(destination))
        else:
            shutil.copytree(figure_dir, destination)
        copied += 1

    action = "Moved" if args.move else "Copied"
    print(f"{action} {copied} folders to {args.target_dir}; skipped {skipped}")


if __name__ == "__main__":
    main()
